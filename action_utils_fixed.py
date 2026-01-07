"""
修复版动作转换工具函数

关键修复：
1. post_process 后的输出是**绝对坐标**，不是相对位移
2. 直接使用绝对坐标作为目标位置，不要加到当前位置上
"""

import numpy as np


def convert_actions_fixed(pred_action, task_type="pick_up_bowl", last_action=None,
                          smoothing_factor=0.0, current_ee_pos=None):
    """
    修复版：正确处理绝对坐标

    Args:
        pred_action: post_process 后的动作 (10维)，前3维是**绝对坐标**（米）
        task_type: 任务类型
        last_action: 上一个动作（用于平滑）
        smoothing_factor: 平滑因子（0=无平滑）
        current_ee_pos: 当前末端执行器位置 (3维: xyz) - 仅用于调试输出
    """
    print("=" * 60)
    print("✅ 使用修复版 convert_actions_fixed")
    print("   模型输出（post_process后）= 绝对坐标")
    print("=" * 60)

    # 1. 基本安全检查
    if np.any(np.isnan(pred_action)) or np.any(np.isinf(pred_action)):
        print(f"⚠️ 检测到NaN或Inf，返回安全位置")
        safe_action = np.array([0.5, 0.0, 0.3, 0.0, 1.0, 0.0, 0.0, 0.08])
        return safe_action

    if len(pred_action) < 3:
        print(f"⚠️ 动作维度不足: {len(pred_action)}，返回安全位置")
        safe_action = np.array([0.5, 0.0, 0.3, 0.0, 1.0, 0.0, 0.0, 0.08])
        return safe_action

    # 2. ✅ 关键修复：前3维是绝对坐标，直接使用
    target_xyz = pred_action[:3].copy()

    print(f"模型预测的目标位置（绝对坐标）: {target_xyz}")
    if current_ee_pos is not None:
        print(f"当前末端执行器位置: {current_ee_pos}")
        delta = target_xyz - current_ee_pos
        print(f"隐含的相对位移: {delta} (范数: {np.linalg.norm(delta):.3f}m)")

    # 3. ✅ 确保目标位置在安全工作空间内（绝对坐标限制）
    # 这是最后的安全保护
    target_xyz[0] = np.clip(target_xyz[0], 0.25, 0.65)   # X: 25-65cm (前后)
    target_xyz[1] = np.clip(target_xyz[1], -0.4, 0.4)   # Y: ±40cm (左右)
    target_xyz[2] = np.clip(target_xyz[2], 0.05, 0.5)   # Z: 5-50cm (上下) ← 降低上限

    print(f"限制后目标位置: x={target_xyz[0]:.3f}, y={target_xyz[1]:.3f}, z={target_xyz[2]:.3f}")

    # 4. 应用平滑（如果需要）
    if last_action is not None and len(last_action) >= 3 and smoothing_factor > 0:
        last_xyz = last_action[:3]
        target_xyz = smoothing_factor * last_xyz + (1 - smoothing_factor) * target_xyz
        print(f"✅ 应用平滑 (α={smoothing_factor}): 新目标位置 = {target_xyz}")

    # 5. 姿态处理
    if len(pred_action) >= 7:
        # 使用模型预测的姿态（四元数）
        quat = pred_action[3:7].copy()
        quat = quat / np.linalg.norm(quat)  # 归一化
        print(f"✅ 使用模型预测的姿态: {quat}")
    else:
        # 固定向下指向的姿态
        quat = np.array([0.0, 1.0, 0.0, 0.0])  # 末端向下
        print(f"✅ 使用固定姿态（末端向下）: {quat}")

    # 6. 夹爪控制
    # 注意: 动作格式为 [x, y, z, rot_6d(6维), gripper(1维)] = 10维
    # 夹爪在索引9 (第10维)
    if len(pred_action) >= 10:
        # 使用模型预测的夹爪宽度
        gripper = pred_action[9]
        # 限制范围
        gripper = np.clip(gripper, 0.0, 0.08)
        print(f"夹爪宽度: {gripper:.4f}m (模型预测)")
    else:
        # 默认打开
        gripper = 0.08
        print(f"夹爪宽度: {gripper:.4f}m (默认打开)")

    # 7. 组合最终动作 [x, y, z, qx, qy, qz, qw, gripper]
    pose_action = np.concatenate([target_xyz, quat, [gripper]])

    print(f"✅ 最终动作（修复版 - 绝对坐标）:")
    print(f"   目标位置: {target_xyz}")
    print(f"   姿态: {quat}")
    print(f"   夹爪: {gripper:.4f}")
    print("=" * 60)

    return pose_action


# 为了兼容性，提供一个包装函数
def convert_actions(pred_action, task_type="pick_up_bowl", last_action=None,
                    smoothing_factor=0.0, current_ee_pos=None):
    """
    包装函数，调用修复版
    """
    return convert_actions_fixed(pred_action, task_type, last_action,
                                  smoothing_factor, current_ee_pos)
