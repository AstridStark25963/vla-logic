import os
import torch
if hasattr(torch, '_dynamo'):
    torch._dynamo.disable()
    print("已禁用 torch._dynamo")
from llava_pythia.conversation import conv_templates, SeparatorStyle
from llava_pythia.model.builder import load_pretrained_model
from llava_pythia.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import torch
from torchvision import transforms
import cv2
from copy import deepcopy
from itertools import repeat
from llava_pythia.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
import numpy as np
import time
import signal
import sys
from aloha_scripts.constants import FPS
from data_utils.datasets import set_seed
from llava_pythia.model import *
from einops import rearrange
import torch_utils as TorchUtils
import matplotlib.pyplot as plt
from collections import deque
# ========== ROS 相关导入 ==========
import rospy
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
from cv_bridge import CvBridge
# =================================

# ========== Franka Panda 关节限制 ==========
FRANKA_JOINT_LIMITS = {
    'min': [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
    'max': [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
}
# ========================================

# 全局变量用于存储deploy_env实例
global global_deploy_env
global_deploy_env = None

def signal_handler(sig, frame):
    """信号处理器，用于安全关闭程序"""
    print('接收到中断信号，正在安全关闭...')
    if global_deploy_env is not None:
        print('执行紧急停止...')
        global_deploy_env.emergency_stop()
    # 确保ROS节点正确关闭
    if not rospy.is_shutdown():
        rospy.signal_shutdown("程序被用户中断")
    print('程序已安全关闭')
    sys.exit(0)
def get_obs(obs, stats):
    images, robot_state = obs
    normalized_state = (robot_state - stats['qpos_mean']) / stats['qpos_std']
    return images, normalized_state
def ensure_quaternion_continuity(current_quat, last_quat):
    """
    确保四元数连续性，避免180度翻转

    Args:
        current_quat: 当前四元数 [x, y, z, w]
        last_quat: 上一个四元数 [x, y, z, w]

    Returns:
        corrected_quat: 修正后的四元数
    """
    if last_quat is None:
        return current_quat

    # 计算点积
    dot_product = np.dot(current_quat, last_quat)

    # 如果点积为负，说明四元数符号不一致
    if dot_product < 0:
        # 取反当前四元数以保持连续性
        return -current_quat

    return current_quat

def convert_actions(pred_action, task_type="pick_up_bowl", last_action=None, smoothing_factor=0.3, current_ee_pos=None):
    """
    改进的动作转换函数 - 将action解释为相对位移而不是绝对坐标

    Args:
        pred_action: 原始预测动作 (10维)
        task_type: 任务类型
        last_action: 上一个动作，用于平滑 (7维: xyz + quat)
        smoothing_factor: 平滑因子 (0-1)，0=无平滑，1=完全使用上一个动作
        current_ee_pos: 当前末端执行器位置 (3维: xyz) - 必需参数
    """
    # 1. 基本检查
    if np.any(np.isnan(pred_action)) or np.any(np.isinf(pred_action)):
        print(f"警告: 检测到NaN或Inf值在预测动作中")
        # 返回零位移而不是固定位置
        safe_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        print(f"使用安全动作(零位移): {safe_action}")
        return safe_action

    # 检查动作维度
    if len(pred_action) < 10:
        print(f"警告: 动作维度不足: {len(pred_action)}，期望至少10维")
        safe_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        return safe_action

    # 检查是否提供了当前位置
    if current_ee_pos is None:
        print("警告: 未提供current_ee_pos，使用默认位置")
        current_ee_pos = np.array([0.3, 0.0, 0.3])

    # 2. 提取分量
    # ✅ 关键修改: pred_action[:3] 现在被解释为相对位移(delta)而不是绝对坐标
    delta_xyz = pred_action[:3].copy()
    cur_rot6d = pred_action[3:9].copy()
    cur_gripper = pred_action[9]

    # 3. 分析原始输出
    print(f"原始位移输出(delta): {delta_xyz}")
    print(f"原始位移幅度: {np.linalg.norm(delta_xyz):.3f}")
    print(f"当前末端执行器位置: {current_ee_pos}")

    # 4. ✅ 限制相对位移的大小（防止单步移动过大）
    # 每步最大位移限制为 5cm
    max_delta = 0.05  # 5cm
    delta_norm = np.linalg.norm(delta_xyz)
    if delta_norm > max_delta:
        print(f"位移过大 ({delta_norm:.3f}m)，限制到 {max_delta}m")
        delta_xyz = delta_xyz / delta_norm * max_delta
        print(f"限制后的位移: {delta_xyz}")

    # 5. ✅ 计算目标位置 = 当前位置 + 相对位移
    target_xyz = current_ee_pos + delta_xyz
    print(f"计算目标位置: {current_ee_pos} + {delta_xyz} = {target_xyz}")

    # 6. 确保目标位置在安全工作空间内（绝对坐标限制）
    # Franka Panda的安全工作空间
    safe_x_range = [0.1, 0.6]    # X范围（防止撞到基座）
    safe_y_range = [-0.4, 0.4]   # Y范围
    safe_z_range = [0.05, 0.7]   # Z范围（防止撞到桌面）

    # 裁剪到安全范围
    for i, (safe_min, safe_max) in enumerate([(safe_x_range[0], safe_x_range[1]),
                                              (safe_y_range[0], safe_y_range[1]),
                                              (safe_z_range[0], safe_z_range[1])]):
        if target_xyz[i] < safe_min:
            print(f"轴{i}低于下界 {safe_min:.3f}，当前值: {target_xyz[i]:.3f}，已修正为: {safe_min:.3f}")
            target_xyz[i] = safe_min
        elif target_xyz[i] > safe_max:
            print(f"轴{i}高于上界 {safe_max:.3f}，当前值: {target_xyz[i]:.3f}，已修正为: {safe_max:.3f}")
            target_xyz[i] = safe_max

    # 7. 应用平滑（如果提供了上一个动作）
    if last_action is not None and len(last_action) >= 3 and smoothing_factor > 0:
        last_xyz = last_action[:3]
        target_xyz = smoothing_factor * last_xyz + (1 - smoothing_factor) * target_xyz
        print(f"应用平滑: 新目标位置 = {target_xyz}")

    # 7. 旋转处理
    try:
        # 7.1 首先归一化旋转6D表示，避免大幅度旋转
        rot6d_norm = np.linalg.norm(cur_rot6d)
        if rot6d_norm > 0:
            # 如果范数太大，归一化到合理范围
            if rot6d_norm > 2.0:
                print(f"旋转6D范数过大 ({rot6d_norm:.3f})，进行归一化")
                cur_rot6d = cur_rot6d / rot6d_norm
                rot6d_norm = 1.0
            elif rot6d_norm < 0.1:
                # 如果范数太小，使用默认旋转（单位矩阵）
                print(f"旋转6D范数过小 ({rot6d_norm:.3f})，使用默认旋转")
                cur_rot6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
                rot6d_norm = np.sqrt(2.0)
        else:
            # 零向量，使用默认旋转
            print("旋转6D为零向量，使用默认旋转")
            cur_rot6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            rot6d_norm = np.sqrt(2.0)

        print(f"归一化后旋转6D范数: {rot6d_norm:.3f}")

        # 7.2 直接从旋转6D转换为旋转矩阵，再转换为四元数
        cur_rot6d_tensor = torch.from_numpy(cur_rot6d).unsqueeze(0).float()
        # 直接转换为旋转矩阵
        rot_matrix = TorchUtils.rotation_6d_to_matrix(cur_rot6d_tensor).squeeze().numpy()

        # 7.3 从旋转矩阵直接转换为四元数
        from scipy.spatial.transform import Rotation as R
        r = R.from_matrix(rot_matrix)
        quat = r.as_quat()

        print(f"原始四元数: {quat}")

        # 7.4 四元数稳定性增强
        # 确保四元数数值稳定性和一致性
        quat_norm = np.linalg.norm(quat)
        if quat_norm > 0:
            quat = quat / quat_norm  # 归一化
        else:
            # 如果四元数为零向量，使用单位四元数
            quat = np.array([0.0, 0.0, 0.0, 1.0])
            print("警告: 四元数为零向量，使用单位四元数")

        # 7.5 应用旋转平滑（如果提供了上一个动作）
        if last_action is not None and len(last_action) >= 7:
            try:
                # 提取上一个动作的四元数
                last_quat = last_action[3:]

                # 检查四元数是否有效
                last_norm = np.linalg.norm(last_quat)
                if last_norm > 0:
                    last_quat = last_quat / last_norm  # 归一化

                    # 使用球面线性插值(Slerp)进行旋转平滑
                    rotation_smoothing = smoothing_factor * 0.3  # 旋转平滑比位置平滑弱
                    if rotation_smoothing > 0:
                        try:
                            last_r = R.from_quat(last_quat)
                            key_rots = R.concatenate([last_r, r])
                            key_times = [0, 1]
                            from scipy.spatial.transform import Slerp
                            slerp = Slerp(key_times, key_rots)
                            smoothed_r = slerp([rotation_smoothing])[0]
                            new_quat = smoothed_r.as_quat()

                            # 确保新四元数与旧四元数方向一致
                            dot_product = np.dot(new_quat, last_quat)
                            if dot_product < 0:
                                new_quat = -new_quat  # 取反以保持连续性

                            quat = new_quat
                            print(f"应用旋转平滑 (Slerp)")
                        except Exception as slerp_error:
                            print(f"Slerp平滑错误: {slerp_error}, 使用线性插值")
                            # 使用简单的线性插值作为备选方案
                            quat = (1 - rotation_smoothing) * quat + rotation_smoothing * last_quat
                            # 归一化结果
                            quat_norm = np.linalg.norm(quat)
                            if quat_norm > 0:
                                quat = quat / quat_norm
                else:
                    print("上一个动作的四元数无效，跳过旋转平滑")
            except Exception as e:
                print(f"旋转平滑错误: {e}, 使用未平滑的四元数")

        # 确保四元数归一化
        quat_norm = np.linalg.norm(quat)
        if abs(quat_norm - 1.0) > 0.01:
            quat = quat / quat_norm
            print(f"四元数已归一化: 范数从 {quat_norm:.3f} 调整到 1.0")

        # 四元数符号处理：避免180度旋转歧义
        # 策略：优先保证连续性，然后尽量保证w分量为正
        quat_modified = False

        # 1. 使用专门的函数确保四元数连续性
        if last_action is not None and len(last_action) >= 7:
            last_quat = last_action[3:]
            original_quat = quat.copy()
            quat = ensure_quaternion_continuity(quat, last_quat)

            if not np.allclose(original_quat, quat):
                quat_modified = True
                print("四元数连续性调整: 已确保符号一致")

        # 2. 在保证连续性的前提下，尽量使w分量为正
        # 注意：如果取反会破坏连续性，则保持原样
        if quat[3] < 0:
            # 检查取反是否会影响连续性
            should_flip = True
            if last_action is not None and len(last_action) >= 7:
                last_quat = last_action[3:]
                dot_if_flipped = np.dot(-quat, last_quat)
                if dot_if_flipped < 0:
                    # 取反会使点积变负，破坏连续性，所以不取反
                    should_flip = False
                    print(f"保持w分量为负以避免破坏连续性 (取反后点积: {dot_if_flipped:.3f})")

            if should_flip:
                quat = -quat
                quat_modified = True
                print("四元数符号规范化: w分量为负，已取反")

        # 3. 额外的180度翻转检测和修正
        # 检查是否发生了接近180度的翻转
        if last_action is not None and len(last_action) >= 7:
            last_quat = last_action[3:]
            # 计算两个四元数之间的角度差
            dot_product = np.abs(np.dot(quat, last_quat))

            # 如果点积接近0，说明两个四元数接近正交，可能发生180度翻转
            if dot_product < 0.1:  # 阈值可以根据实际情况调整
                print(f"检测到可能的180度翻转，点积: {dot_product:.3f}")

                # 尝试多种修正方法，选择最合适的
                candidates = [
                    quat,           # 原始四元数
                    -quat,          # 取反
                ]

                best_candidate = quat
                best_dot = dot_product

                for candidate in candidates:
                    candidate_dot = np.abs(np.dot(candidate, last_quat))
                    if candidate_dot > best_dot:
                        best_dot = candidate_dot
                        best_candidate = candidate

                if not np.allclose(best_candidate, quat):
                    quat = best_candidate
                    quat_modified = True
                    print(f"已修正180度翻转，新的点积: {best_dot:.3f}")

        if quat_modified:
            # 重新归一化（取反不影响范数，但为了安全）
            quat_norm = np.linalg.norm(quat)
            if abs(quat_norm - 1.0) > 0.01:
                quat = quat / quat_norm

        # 8. 姿态合理性检查和修正（针对抓取任务）
        # 检查是否为抓取碗的任务，如果是则确保末端执行器向下指向
        if "pick" in task_type.lower() or "bowl" in task_type.lower():
            print("检测到抓取任务，检查姿态合理性...")

            # 将四元数转换为欧拉角进行分析
            r_current = R.from_quat(quat)
            euler_current = r_current.as_euler('xyz', degrees=True)

            print(f"当前欧拉角: roll={euler_current[0]:.2f}°, pitch={euler_current[1]:.2f}°, yaw={euler_current[2]:.2f}°")

            # 检查Z轴方向是否向下（适合抓取）
            z_direction = r_current.apply([0, 0, 1])  # 应用旋转到原始Z轴
            print(f"当前Z轴方向: [{z_direction[0]:.3f}, {z_direction[1]:.3f}, {z_direction[2]:.3f}]")

            # 如果Z轴不是向下指向（Z分量应该接近-1）
            if z_direction[2] > -0.5:  # 如果不是明显向下指向
                print(f"姿态不合理: Z轴方向 {z_direction[2]:.3f} 不适合抓取任务")

                # 使用确定的向下指向姿态 [0, 1, 0, 0] (绕Y轴180度)
                # 这个姿态确保末端执行器Z轴向下指向[0, 0, -1]，避免右后方旋转问题
                print("检测到姿态不合理，强制设置为标准向下指向姿态")
                quat = np.array([0.0, 1.0, 0.0, 0.0])

                # 验证修正效果
                r_fixed = R.from_quat(quat)
                fixed_z_direction = r_fixed.apply([0, 0, 1])
                fixed_x_direction = r_fixed.apply([1, 0, 0])
                fixed_y_direction = r_fixed.apply([0, 1, 0])

                print(f"修正后四元数: {quat}")
                print(f"修正后Z轴方向: [{fixed_z_direction[0]:.3f}, {fixed_z_direction[1]:.3f}, {fixed_z_direction[2]:.3f}] (应为[0,0,-1])")
                print(f"修正后X轴方向: [{fixed_x_direction[0]:.3f}, {fixed_x_direction[1]:.3f}, {fixed_x_direction[2]:.3f}] (应为[-1,0,0])")
                print(f"修正后Y轴方向: [{fixed_y_direction[0]:.3f}, {fixed_y_direction[1]:.3f}, {fixed_y_direction[2]:.3f}] (应为[0,1,0])")
                print("✅ 修正完成，末端执行器现在应该向下指向")

        # 9. 夹爪处理
        # 使用sigmoid函数将夹爪值映射到[0,1]，更平滑
        gripper_value = 1.0 / (1.0 + np.exp(-cur_gripper))
        print(f"原始夹爪值: {cur_gripper:.3f}, sigmoid处理后: {gripper_value:.3f}")

        # 10. 组合最终动作
        # 将夹爪值转换为物理单位（米），Franka夹爪最大开合距离约为0.08m
        gripper_width = gripper_value * 0.08
        # ✅ 使用计算后的目标位置而不是原始的cur_xyz
        pose_action = np.concatenate((target_xyz, quat, [gripper_width]))

        print(f"改进转换结果(相对位移模式):")
        print(f"  当前位置: {current_ee_pos}")
        print(f"  相对位移: {delta_xyz}")
        print(f"  目标位置: {target_xyz}")
        print(f"  四元数: {quat}")
        print(f"  四元数范数: {np.linalg.norm(quat):.3f}")
        print(f"  夹爪值: {gripper_value:.3f}")
        print(f"  夹爪宽度: {gripper_width:.3f}m")
        print(f"  动作维度: {len(pose_action)}, 动作内容: {pose_action}")

        # 11. 最终安全检查
        # 确保位置在绝对安全范围内
        pose_action[0] = np.clip(pose_action[0], 0.1, 0.6)   # X轴
        pose_action[1] = np.clip(pose_action[1], -0.4, 0.4)  # Y轴
        pose_action[2] = np.clip(pose_action[2], 0.05, 0.7)  # Z轴

        # 确保四元数有效
        if len(pose_action) >= 7:
            quat_norm = np.linalg.norm(pose_action[3:7])
            if quat_norm > 0:
                pose_action[3:7] = pose_action[3:7] / quat_norm
            else:
                pose_action[3:7] = np.array([0.0, 0.0, 0.0, 1.0])

        # 确保夹爪值在合理范围内
        # 注意: 动作格式为 [x, y, z, rot_6d(6维), gripper(1维)] = 10维
        # 夹爪在索引9 (第10维)
        if len(pose_action) >= 10:
            pose_action[9] = np.clip(pose_action[9], 0.0, 0.08)

        return pose_action

    except Exception as e:
        print(f"旋转转换错误: {e}")
        safe_action = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0])
        return safe_action
class llava_pythia_act_policy:
    def __init__(self, policy_config, data_args=None):
        super().__init__()
        self.load_policy(policy_config)
        self.data_args = data_args
    def load_policy(self, policy_config):
        # 1. 保存 policy_config
        self.policy_config = policy_config
        from transformers import AutoTokenizer, GPTNeoXTokenizerFast, AutoConfig
        from llava_pythia.model import LlavaPythiaForCausalLM
        from peft import PeftModel
        import os
        import torch
        from torch.nn.functional import interpolate
        model_base = policy_config["model_base"]
        model_path = policy_config["model_path"]
        print("正在加载基础模型...")
        # 2. 加载 tokenizer
        try:
            self.tokenizer = GPTNeoXTokenizerFast.from_pretrained(model_base)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # 3. 加载基础模型
        base_model = LlavaPythiaForCausalLM.from_pretrained(
            model_base,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
            device_map=None,
            trust_remote_code=True
        ).cpu()
        print("正在加载 LoRA 权重...")
        self.policy = PeftModel.from_pretrained(base_model, model_path, is_trainable=False)
        self.policy = self.policy.merge_and_unload()
        self.policy = self.policy.cuda()
        # 4. 修复位置编码（320x320）
        def interpolate_pos_encoding(vision_tower, image_size=320, patch_size=14):
            vision_model = vision_tower.vision_model
            old_pos_embed = vision_model.embeddings.position_embedding.weight.data
            cls_pos_embed = old_pos_embed[0:1]
            patch_pos_embed = old_pos_embed[1:]
            h = w = image_size // patch_size
            new_hw = h * w
            embed_dim = patch_pos_embed.shape[1]
            patch_pos_embed = patch_pos_embed.transpose(0, 1).view(1, embed_dim, 24, 24)
            new_patch_pos_embed = interpolate(
                patch_pos_embed, size=(h, w), mode='bicubic', align_corners=False
            )
            new_patch_pos_embed = new_patch_pos_embed.view(embed_dim, new_hw).transpose(0, 1)
            new_pos_embed = torch.cat([cls_pos_embed, new_patch_pos_embed], dim=0)
            vision_model.embeddings.position_embedding = torch.nn.Embedding.from_pretrained(new_pos_embed, freeze=False)
            vision_model.embeddings.position_ids = torch.arange(new_pos_embed.shape[0]).expand((1, -1)).cuda()
            print(f"位置编码已从 577 → {new_pos_embed.shape[0]}（适配 {image_size}x{image_size}）")
        if hasattr(self.policy, 'get_vision_tower') and self.policy.get_vision_tower() is not None:
            interpolate_pos_encoding(self.policy.get_vision_tower(), image_size=320, patch_size=14)
        # 5. ✅✅✅ 关键修复：加载 non_lora_trainables.bin，正确处理 key + 跳过位置编码 + 清理 NaN
        non_lora_path = os.path.join(model_path, 'non_lora_trainables.bin')
        if os.path.exists(non_lora_path):
            print("正在加载 non_lora_trainables.bin...")
            non_lora_weights = torch.load(non_lora_path, map_location='cpu')
            cleaned_weights = {}
            for k, v in non_lora_weights.items():
                # 🔥 移除 'base_model.model.' 前缀（根据你日志中的 key）
                if k.startswith('base_model.model.'):
                    k = k[len('base_model.model.'):]
                # 🔥 跳过位置编码（避免与插值后冲突）
                if 'vision_model.embeddings.position_embedding.weight' in k:
                    print(f"跳过位置编码权重: {k}")
                    continue
                # 🔥 清理 NaN / Inf
                if torch.isnan(v).any() or torch.isinf(v).any():
                    print(f"⚠️ 权重 {k} 包含 NaN/Inf，清理中...")
                    v = torch.nan_to_num(v, nan=0.0, posinf=1e4, neginf=-1e4)
                # 🔥 强制 float32
                if v.dtype == torch.float16:
                    v = v.float()
                cleaned_weights[k] = v
            missing, unexpected = self.policy.load_state_dict(cleaned_weights, strict=False)
            print(f"non_lora_trainables 加载完成！missing={len(missing)}, unexpected={len(unexpected)}")
        else:
            print("警告: 未找到 non_lora_trainables.bin")
        # 6. 修复 config.concat
        trained_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.policy.config = trained_config
        self.config = trained_config
        if not hasattr(self.config, 'concat') or self.config.concat is None:
            self.config.concat = "token_cat"
            self.policy.config.concat = "token_cat"
        self.policy.visual_concat = getattr(self.policy.config, 'concat', 'token_cat')
        print(f"修复后: 模型visual_concat属性 = {self.policy.visual_concat}")
        # 7. 强制 float32 + 检查 NaN
        self.policy = self.policy.to(torch.float32)
        has_nan = False
        for name, param in self.policy.named_parameters():
            if torch.isnan(param).any():
                print(f"❌ 修复后仍存在 NaN: {name}")
                has_nan = True
        if not has_nan:
            print("✅ 模型参数中未检测到 NaN")
        # 8. 打印模型设备
        print(f"模型已加载到设备: {next(self.policy.parameters()).device}")
        # 9. 初始化 image_processor
        from transformers import CLIPImageProcessor
        self.image_processor = CLIPImageProcessor.from_pretrained(
            model_base,
            size={"height": 320, "width": 320},
            do_center_crop=False,
            do_normalize=True,
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
        )
        self.context_len = 2048
        print("策略模型加载完成（含 NaN 修复 + 位置编码适配）")
    def _fix_vision_tower_config(self):
        """修复视觉塔配置，确保使用正确的图像尺寸"""
        if hasattr(self.policy, 'get_vision_tower') and self.policy.get_vision_tower() is not None:
            vision_tower = self.policy.get_vision_tower()
            if hasattr(vision_tower, 'vision_tower'):
                vision_model = vision_tower.vision_tower
                # 设置图像尺寸为320
                if hasattr(vision_model, 'config'):
                    vision_model.config.image_size = 320
                if hasattr(vision_model, 'vision_model') and hasattr(vision_model.vision_model, 'config'):
                    vision_model.vision_model.config.image_size = 320
                print("视觉塔配置已修复为320x320")
    def _unify_dtypes(self):
        """统一模型的数据类型"""
        # 检查模型当前的数据类型
        current_dtype = next(self.policy.parameters()).dtype
        print(f"模型当前数据类型: {current_dtype}")
        # 在CPU环境下强制使用float32以避免half精度问题
        if not torch.cuda.is_available() and current_dtype == torch.float16:
            print("检测到CPU环境，强制将模型转换为float32以避免half精度问题")
            self.policy = self.policy.to(torch.float32)
            current_dtype = torch.float32
        # 确保所有组件使用相同的数据类型
        if hasattr(self.policy, 'get_vision_tower') and self.policy.get_vision_tower() is not None:
            vision_tower = self.policy.get_vision_tower()
            vision_tower.to(dtype=current_dtype)
        # 确保动作头使用相同的数据类型
        if hasattr(self.policy, 'action_head'):
            self.policy.action_head.to(dtype=current_dtype)
        print(f"模型数据类型已统一为: {current_dtype}")
    def manual_load_components(self, model_path, model_base):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from llava_pythia.model import LlavaPythiaForCausalLM
        from llava_pythia import LlavaPythiaConfig
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_base or model_path, use_fast=False)
        except:
            from transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_base or model_path)
        self.config = LlavaPythiaConfig.from_pretrained(model_path)
        # 关键修复：在加载模型前设置正确的图像尺寸
        if hasattr(self.config, 'vision_tower'):
            self.config.vision_tower.image_size = 320
        # 使用float32避免数据类型不匹配
        self.policy = LlavaPythiaForCausalLM.from_pretrained(
            model_path,
            config=self.config,
            torch_dtype=torch.float32,  # 使用float32确保一致性
            low_cpu_mem_usage=True
        ).cuda()
        # 强制设置视觉编码器的图像尺寸
        self._fix_vision_tower_config()
        # 统一数据类型
        self._unify_dtypes()
        from llava_pythia.mm_utils import get_model_name_from_path
        model_name = get_model_name_from_path(model_path)
        if 'llava' in model_name.lower():
            from transformers import CLIPImageProcessor
            # 明确指定320x320尺寸
            self.image_processor = CLIPImageProcessor.from_pretrained(
                model_base or model_path,
                do_resize=True,
                size={"height": 320, "width": 320},
                do_center_crop=False,  # 禁用中心裁剪，使用resize
                do_normalize=True,
                image_mean=[0.48145466, 0.4578275, 0.40821073],
                image_std=[0.26862954, 0.26130258, 0.27577711],
            )
        else:
            from torchvision import transforms
            self.image_processor = transforms.Compose([
                transforms.Resize((320, 320), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])
        self.context_len = 2048
        # >>>>>>>>>>>>>>>>>>> 新增：加载 non_lora_trainables.bin <<<<<<<<<<<<<<<<<<<<<
        import os
        non_lora_path = os.path.join(model_path, 'non_lora_trainables.bin')
        if os.path.exists(non_lora_path):
            print("正在加载 non_lora_trainables.bin (manual mode)...")
            non_lora_weights = torch.load(non_lora_path, map_location='cpu')
            new_weights = {}
            for k, v in non_lora_weights.items():
                if k.startswith('model.'):
                    new_weights[k] = v
                else:
                    new_weights['model.' + k] = v
            missing, unexpected = self.policy.load_state_dict(new_weights, strict=False)
            print(f"non_lora_trainables 加载完成（手动模式）！")
            if missing:
                print(f"Missing keys: {missing}")
            if unexpected:
                print(f"Unexpected keys: {unexpected}")
        else:
            print("警告: 未找到 non_lora_trainables.bin，路径:", non_lora_path)
        # >>>>>>>>>>>>>>>>>>> 新增结束 <<<<<<<<<<<<<<<<<<<<<
    def process_batch_to_llava(self, curr_image, robo_state, raw_lang):
        """
        适配 image_size=320 的模型。
        curr_image: (2, H, W, 3) in [0, 1]
        """
        self.conv = conv_templates[self.policy_config['conv_mode']].copy()
        if len(curr_image.shape) == 5:
            curr_image = curr_image.squeeze(0)  # (2, H, W, 3)
        assert curr_image.dim() == 4 and curr_image.shape[0] == 2, f"curr_image shape: {curr_image.shape}"
        # 转为 (2, 3, H, W)
        curr_image = curr_image.permute(0, 3, 1, 2)
        
        # >>>>>>>>>> 新增：保存输入图像到 /home/tianxiaoyan/Pictures <<<<<<<<<<
        import os
        import cv2
        output_dir = "/home/tianxiaoyan/Pictures/camera_inputs"
        os.makedirs(output_dir, exist_ok=True)
        
        for i in range(curr_image.shape[0]):
            # 将图像从 [0,1] 范围转换到 [0,255] 且转换为 uint8
            img_save = (curr_image[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            # 转换为BGR格式保存（OpenCV需要）
            if img_save.shape[2] == 3:  # RGB image
                img_save = cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR)
            output_path = os.path.join(output_dir, f"camera_input_{i}.jpg")
            cv2.imwrite(output_path, img_save)
        print(f"已保存输入图像到 {output_dir}")
        # >>>>>>>>>> 结束新增 <<<<<<<<<<
        
        # 获取模型的数据类型
        model_dtype = next(self.policy.parameters()).dtype
        # 统一的图像预处理 - 确保输出为320x320
        processed_images = []
        for i in range(curr_image.shape[0]):
            img = curr_image[i]  # (3, H, W)
            # === 优化：纯 GPU 预处理，避免 CPU-GPU 拷贝 ===
            # 1. 调整尺寸到 320x320
            img_resized = torch.nn.functional.interpolate(
                img.unsqueeze(0), size=(320, 320), mode='bilinear', align_corners=False
            ).squeeze(0)  # (3, 320, 320)
            # 2. 归一化（使用与 CLIP 相同的 mean/std）
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=img.device).view(3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=img.device).view(3, 1, 1)
            img_norm = (img_resized - mean) / std
            processed_images.append(img_norm.unsqueeze(0).to(dtype=model_dtype))
        # 合并
        image_tensor = torch.cat(processed_images, dim=0).to(self.policy.device)
        # === 优化结束 ===
        image_tensor_main = image_tensor[0:1]  # 主视角
        image_tensor_secondary = image_tensor[1:2]  # 副视角
        inp = raw_lang
        if self.policy.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        self.conv.append_message(self.conv.roles[0], inp)
        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt() + "  "
        # 确保input_ids使用正确设备
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        attn_mask = input_ids.ne(self.tokenizer.pad_token_id)
        # 确保状态使用相同的数据类型和设备
        states = robo_state.to(self.policy.device, dtype=model_dtype)
        return dict(
            input_ids=input_ids,
            attention_mask=attn_mask,
            images=image_tensor_main,
            images_r=image_tensor_secondary,
            states=states
        )
class FrankaROSEnvironment:
    def __init__(self, left_cam_id=4, right_cam_id=10):
        self.left_cam_id = left_cam_id
        self.right_cam_id = right_cam_id
        self.left_cap = None
        self.right_cap = None
        self.joint_positions = np.zeros(7)  # 存储当前关节位置
        self.current_ee_pose = None  # 存储当前末端执行器位姿
        self.init_cameras()
        if not rospy.get_node_uri():
            rospy.init_node('tinyvla_franka_control', anonymous=True)
        self.pose_pub = rospy.Publisher(
            '/cartesian_impedance_example_controller/equilibrium_pose',
            PoseStamped,
            queue_size=1
        )
        # 添加夹爪控制发布者
        self.gripper_pub = rospy.Publisher(
            '/franka_gripper/goal_width',
            Float64,
            queue_size=1
        )
        # 订阅关节状态
        self.joint_sub = rospy.Subscriber(
            '/franka_state_controller/joint_states',
            JointState,
            self.joint_state_callback,
            queue_size=1
        )
        # 订阅末端执行器位姿
        self.ee_pose_sub = rospy.Subscriber(
            '/cartesian_impedance_example_controller/equilibrium_pose',
            PoseStamped,
            self.ee_pose_callback,
            queue_size=1
        )
        rospy.sleep(1.0)
        print("Franka 环境初始化完成（笛卡尔控制模式）")

    def check_joint_safety(self, target_joints):
        """检查目标关节角度是否在安全范围内"""
        for i in range(min(len(target_joints), 7)):
            if target_joints[i] < FRANKA_JOINT_LIMITS['min'][i] or target_joints[i] > FRANKA_JOINT_LIMITS['max'][i]:
                print(f"警告: 关节 {i+1} 超出安全范围! 当前值: {target_joints[i]:.3f}, 安全范围: [{FRANKA_JOINT_LIMITS['min'][i]:.3f}, {FRANKA_JOINT_LIMITS['max'][i]:.3f}]")
                return False
        return True

    def emergency_stop(self):
        """紧急停止机械臂"""
        print("执行紧急停止!")
        # 发送当前位置作为目标，使机械臂停止移动
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "panda_link0"
        msg.pose.position.x = self.joint_positions[0] if len(self.joint_positions) > 0 else 0.0
        msg.pose.position.y = self.joint_positions[1] if len(self.joint_positions) > 1 else 0.0
        msg.pose.position.z = self.joint_positions[2] if len(self.joint_positions) > 2 else 0.3
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0
        self.pose_pub.publish(msg)

        # 发送夹爪停止命令
        gripper_msg = Float64()
        gripper_msg.data = 0.08  # 设置为最大开合
        self.gripper_pub.publish(gripper_msg)

        print("紧急停止命令已发送")

    def joint_state_callback(self, msg):
        """关节状态回调函数"""
        # 提取前7个关节的角度
        if len(msg.position) >= 7:
            self.joint_positions = np.array(msg.position[:7])

    def ee_pose_callback(self, msg):
        """末端执行器位姿回调函数"""
        # 存储当前末端执行器位姿
        self.current_ee_pose = msg

    def get_current_ee_position(self):
        """获取当前末端执行器位置"""
        if self.current_ee_pose is None:
            # 如果还没有接收到位姿，返回默认安全位置
            print("警告: 还未接收到末端执行器位姿，使用默认位置")
            return np.array([0.3, 0.0, 0.3])

        return np.array([
            self.current_ee_pose.pose.position.x,
            self.current_ee_pose.pose.position.y,
            self.current_ee_pose.pose.position.z
        ])

    def reset(self, randomize=False):
        print("环境重置")
        # 在重置时发送一个安全的初始位置
        self.send_safe_position()
        return self.get_observation()

    def send_safe_position(self):
        """发送一个安全的初始位置，确保夹爪向下指向且朝向正确"""
        print("发送安全初始位置...")
        # 安全位置：X=0.3, Y=0.0, Z=0.3（在基座前方，适当高度）
        # 使用修正后的四元数 [0.0, 1.0, 0.0, 0.0] (绕Y轴180度)
        # 这个姿态确保:
        # - Z轴向下指向 [0, 0, -1]
        # - X轴向后指向 [-1, 0, 0] (与动作修正逻辑一致，避免水平180度旋转)
        # - Y轴向右指向 [0, 1, 0]
        # 在ROS中，四元数顺序是 [x, y, z, w]
        safe_position = [0.3, 0.0, 0.3, 0.0, 1.0, 0.0, 0.0]  # x, y, z, qx, qy, qz, qw

        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "panda_link0"
        msg.pose.position.x = safe_position[0]
        msg.pose.position.y = safe_position[1]
        msg.pose.position.z = safe_position[2]
        msg.pose.orientation.x = safe_position[3]
        msg.pose.orientation.y = safe_position[4]
        msg.pose.orientation.z = safe_position[5]
        msg.pose.orientation.w = safe_position[6]

        print(f"发送安全位置: x={safe_position[0]}, y={safe_position[1]}, z={safe_position[2]}")
        print("确保夹爪向下指向且朝向正确...")
        self.pose_pub.publish(msg)
        rospy.sleep(2.0)  # 等待位置发送完成

        # 验证姿态
        print("验证安全姿态:")
        print(f"  位置: x={safe_position[0]}, y={safe_position[1]}, z={safe_position[2]}")
        print(f"  四元数: [{safe_position[3]}, {safe_position[4]}, {safe_position[5]}, {safe_position[6]}]")
        print("  ✅ 夹爪应该向下指向且朝向正确")

        # 验证四元数是否正确
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat([safe_position[3], safe_position[4], safe_position[5], safe_position[6]])
        z_axis = r.apply([0, 0, 1])
        x_axis = r.apply([1, 0, 0])
        y_axis = r.apply([0, 1, 0])

        print(f"  验证Z轴: [{z_axis[0]:.3f}, {z_axis[1]:.3f}, {z_axis[2]:.3f}] {'✅向下' if z_axis[2] < -0.9 else '❌不向下'}")
        print(f"  验证X轴: [{x_axis[0]:.3f}, {x_axis[1]:.3f}, {x_axis[2]:.3f}] {'✅向后' if x_axis[0] < -0.9 else '❌'}")
        print(f"  验证Y轴: [{y_axis[0]:.3f}, {y_axis[1]:.3f}, {y_axis[2]:.3f}] {'✅向右' if y_axis[1] > 0.9 else '❌'}")

        # 检查是否与动作修正逻辑一致（避免水平旋转180度的问题）
        if x_axis[0] < -0.9 and y_axis[1] > 0.9:  # X轴向后，Y轴向右
            print("  ✅ 姿态与动作修正逻辑一致，避免了水平旋转180度问题")
        else:
            print("  ❌ 姿态与动作修正逻辑不一致，可能导致旋转问题")
    def init_cameras(self):
        print("初始化RealSense摄像头...")
        self.left_cap = cv2.VideoCapture(self.left_cam_id)
        self.right_cap = cv2.VideoCapture(self.right_cam_id)
        for cap, name in [(self.left_cap, "左"), (self.right_cap, "右")]:
            if not cap.isOpened():
                print(f"错误: 无法打开{name}摄像头")
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("摄像头初始化完成")
    def get_images(self):
        left_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        right_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        if self.left_cap and self.left_cap.isOpened():
            ret, img = self.left_cap.read()
            if ret: left_image = img
        if self.right_cap and self.right_cap.isOpened():
            ret, img = self.right_cap.read()
            if ret: right_image = img
        return left_image, right_image
    def get_observation(self):
        left_img, right_img = self.get_images()
        if left_img.shape != right_img.shape:
            min_h = min(left_img.shape[0], right_img.shape[0])
            min_w = min(left_img.shape[1], right_img.shape[1])
            left_img = cv2.resize(left_img, (min_w, min_h))
            right_img = cv2.resize(right_img, (min_w, min_h))
        images = np.stack([left_img, right_img], axis=0)
        state = np.zeros(7)
        return images, state
    def step(self, action):
        print(f"接收到动作: {action}")
        print(f"动作维度: {len(action)}")
        print(f"发布位姿: pos=({action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f})")

        # 检查末端执行器位置是否在安全范围内
        safety_check_passed = True
        safety_violations = []

        # 更严格的X轴检查（防止朝向基座）
        # X轴正值表示向前，负值表示向后（朝向基座）
        if action[0] < 0.1 or action[0] > 0.6:
            safety_violations.append(f"X轴超出安全范围: {action[0]:.3f} (建议范围: 0.1-0.6)")
            safety_check_passed = False

        # Y轴检查（左右移动）
        if action[1] < -0.4 or action[1] > 0.4:
            safety_violations.append(f"Y轴超出安全范围: {action[1]:.3f}")
            safety_check_passed = False

        # Z轴检查（上下移动）
        if action[2] < 0.05 or action[2] > 0.7:
            safety_violations.append(f"Z轴超出安全范围: {action[2]:.3f} (建议范围: 0.05-0.7)")
            safety_check_passed = False

        # 特殊检查：防止过于靠近基座
        distance_from_base = np.sqrt(action[0]**2 + action[1]**2)
        if distance_from_base < 0.15:
            safety_violations.append(f"距离基座过近: {distance_from_base:.3f}m")
            safety_check_passed = False

        if not safety_check_passed:
            print("安全检查失败:")
            for violation in safety_violations:
                print(f"  - {violation}")
            print("使用安全位置替代危险动作")

            # 使用安全位置而不是简单裁剪
            safe_action = action.copy()
            # 确保X轴在安全范围内（防止朝向基座）
            safe_action[0] = max(0.15, min(0.5, action[0]))
            # 确保Y轴在安全范围内
            safe_action[1] = max(-0.3, min(0.3, action[1]))
            # 确保Z轴在安全范围内
            safe_action[2] = max(0.1, min(0.6, action[2]))

            # 确保不会过于靠近基座中心
            distance = np.sqrt(safe_action[0]**2 + safe_action[1]**2)
            if distance < 0.15:
                # 调整X值以确保足够的距离
                safe_action[0] = 0.15 if safe_action[0] >= 0 else -0.15

            action = safe_action
            print(f"修正后的位置: pos=({action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f})")

        # 额外的姿态安全性检查
        if len(action) >= 7:
            # 检查四元数是否有效
            quat = action[3:7]
            quat_norm = np.linalg.norm(quat)

            if abs(quat_norm - 1.0) > 0.1:  # 如果四元数范数偏离单位四元数太多
                print(f"警告: 四元数范数异常 ({quat_norm:.3f})，使用默认姿态")
                # 使用确定的向下指向姿态 [0, 1, 0, 0] (绕Y轴180度)
                # 这个姿态确保末端执行器Z轴向下指向[0, 0, -1]，避免右后方旋转问题
                print("❌ 四元数异常，使用确定的安全向下指向姿态 [0, 1, 0, 0]")
                action[3:7] = np.array([0.0, 1.0, 0.0, 0.0])  # 向下指向的四元数

                # 验证安全姿态
                r_safe = R.from_quat(action[3:7])
                safe_z_direction = r_safe.apply([0, 0, 1])
                safe_x_direction = r_safe.apply([1, 0, 0])
                safe_y_direction = r_safe.apply([0, 1, 0])

                print(f"✅ 安全姿态验证:")
                print(f"   四元数: {action[3:7]}")
                print(f"   Z轴: [{safe_z_direction[0]:.3f}, {safe_z_direction[1]:.3f}, {safe_z_direction[2]:.3f}] (应为[0,0,-1])")
                print(f"   X轴: [{safe_x_direction[0]:.3f}, {safe_x_direction[1]:.3f}, {safe_x_direction[2]:.3f}] (应为[-1,0,0])")
                print(f"   Y轴: [{safe_y_direction[0]:.3f}, {safe_y_direction[1]:.3f}, {safe_y_direction[2]:.3f}] (应为[0,1,0])")

            # 检查姿态是否合理（避免极端姿态）
            try:
                from scipy.spatial.transform import Rotation as R
                r = R.from_quat(quat)
                euler = r.as_euler('xyz', degrees=True)

                print(f"当前姿态欧拉角: roll={euler[0]:.2f}°, pitch={euler[1]:.2f}°, yaw={euler[2]:.2f}°")

                # 检查Z轴方向是否向下（适合抓取）
                z_direction = r.apply([0, 0, 1])  # 应用旋转到原始Z轴
                print(f"当前Z轴方向: [{z_direction[0]:.3f}, {z_direction[1]:.3f}, {z_direction[2]:.3f}]")

                # 如果Z轴不是向下指向（Z分量应该接近-1）
                if z_direction[2] > -0.5:  # 如果不是明显向下指向
                    print(f"警告: Z轴方向异常 ({z_direction[2]:.3f})，调整为向下指向姿态")
                    # 使用确定的向下指向姿态 [0, 1, 0, 0] (绕Y轴180度)
                    # 这个姿态确保末端执行器Z轴向下指向[0, 0, -1]，避免右后方旋转问题
                    print("⚠️ 安全检查: 检测到姿态异常，强制设置为标准向下指向姿态")
                    action[3:7] = np.array([0.0, 1.0, 0.0, 0.0])

                    # 验证修正效果
                    r_fixed = R.from_quat(action[3:7])
                    fixed_z_direction = r_fixed.apply([0, 0, 1])
                    fixed_x_direction = r_fixed.apply([1, 0, 0])
                    fixed_y_direction = r_fixed.apply([0, 1, 0])

                    print(f"✅ 安全修正完成:")
                    print(f"   修正后四元数: {action[3:7]}")
                    print(f"   Z轴方向: [{fixed_z_direction[0]:.3f}, {fixed_z_direction[1]:.3f}, {fixed_z_direction[2]:.3f}] (应为[0,0,-1])")
                    print(f"   X轴方向: [{fixed_x_direction[0]:.3f}, {fixed_x_direction[1]:.3f}, {fixed_x_direction[2]:.3f}] (应为[-1,0,0])")
                    print(f"   Y轴方向: [{fixed_y_direction[0]:.3f}, {fixed_y_direction[1]:.3f}, {fixed_y_direction[2]:.3f}] (应为[0,1,0])")

            except Exception as e:
                print(f"姿态检查错误: {e}")
                # 出错时使用安全姿态
                action[3:7] = np.array([0.0, 1.0, 0.0, 0.0])

        # 发布笛卡尔位姿
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "panda_link0"
        msg.pose.position.x = action[0]
        msg.pose.position.y = action[1]
        msg.pose.position.z = action[2]
        msg.pose.orientation.x = action[3]
        msg.pose.orientation.y = action[4]
        msg.pose.orientation.z = action[5]
        msg.pose.orientation.w = action[6]
        print("发布的位姿消息:")
        print(msg)
        self.pose_pub.publish(msg)

        # 发布夹爪控制命令（如果动作包含夹爪控制）
        # 注意: 动作格式为 [x, y, z, rot_6d(6维), gripper(1维)] = 10维
        # 夹爪在索引9 (第10维)
        if len(action) >= 10:
            # 夹爪控制值已经在convert_actions函数中转换为物理单位（米）
            gripper_width = action[9]  # 第10个元素是夹爪宽度（米）
            # 限制夹爪宽度在合理范围内
            gripper_width = max(0.0, min(0.08, gripper_width))
            gripper_msg = Float64()
            gripper_msg.data = gripper_width
            print(f"发布夹爪控制命令: {gripper_width:.3f}m")
            self.gripper_pub.publish(gripper_msg)
        else:
            print("动作不包含夹爪控制信息")

        return self.get_observation(), False, {}
    def __del__(self):
        if self.left_cap: self.left_cap.release()
        if self.right_cap: self.right_cap.release()
def debug_model_outputs(policy, batch, step):
    """调试模型输出"""
    print(f"\n=== 步骤 {step} 调试信息 ===")
    # 检查输入数据
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"输入 {key}: shape={value.shape}, range=[{value.min():.3f}, {value.max():.3f}], "
                  f"NaN={torch.isnan(value).any()}, Inf={torch.isinf(value).any()}")
    # 检查模型参数
    model_params = list(policy.policy.parameters())
    if model_params:
        first_param = model_params[0]
        print(f"模型参数: shape={first_param.shape}, range=[{first_param.min():.3f}, {first_param.max():.3f}], "
              f"NaN={torch.isnan(first_param).any()}, Inf={torch.isinf(first_param).any()}")
    print("=== 调试结束 ===\n")


def test_quaternion_continuity():
    """测试四元数连续性处理函数"""
    print("测试四元数连续性处理...")

    # 测试用例1: 正常情况
    q1 = np.array([0.0, 0.0, 0.0, 1.0])  # 单位四元数
    q2 = np.array([0.1, 0.0, 0.0, 0.995])  # 接近单位四元数
    result = ensure_quaternion_continuity(q2, q1)
    print(f"测试1 - 原始: {q2}, 结果: {result}, 是否相等: {np.allclose(q2, result)}")

    # 测试用例2: 符号相反的情况
    q1 = np.array([0.0, 0.0, 0.0, 1.0])
    q2 = np.array([0.0, 0.0, 0.0, -1.0])  # 相反符号
    result = ensure_quaternion_continuity(q2, q1)
    print(f"测试2 - 原始: {q2}, 结果: {result}, 是否取反: {np.allclose(result, -q2)}")

    print("四元数连续性测试完成")
def eval_bc(policy, deploy_env, policy_config, save_episode=True, num_rollouts=1, raw_lang=None):
    assert raw_lang is not None, "raw lang is None!!!!!!"
    set_seed(0)
    rand_crop_resize = False
    temporal_agg = True
    action_dim = policy.config.action_dim
    policy.policy.eval()
    import pickle
    stats_path = os.path.join(policy_config['model_path'], 'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    env = deploy_env
    query_frequency = policy.config.chunk_size / 2
    if temporal_agg:
        query_frequency = 1
        num_queries = policy.config.chunk_size
    max_timesteps = int(10000)
    # 优化终止条件
    max_duration = 60  # 增加最大运行时间到60秒
    target_reached_threshold = 0.01  # 降低位置阈值，提高精度
    min_steps_for_completion = 100  # 至少执行100步才考虑终止
    for rollout_id in range(num_rollouts):
        env.reset(randomize=False)
        print(f"env has reset!")
        model_dtype = next(policy.policy.parameters()).dtype
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, action_dim], dtype=model_dtype).cuda()
        image_list = []
        robot_state_list = []
        target_action_list = []
        start_time = time.time()
        last_position = None
        last_action = None  # 添加上一个动作记录，用于平滑
        stationary_count = 0
        success_count = 0
        with torch.inference_mode():
            time0 = time.time()
            DT = 1 / FPS
            culmulated_delay = 0
            for t in range(max_timesteps):
                # 检查运行时间限制
                current_time = time.time()
                if current_time - start_time > max_duration:
                    print(f"达到最大运行时间 {max_duration} 秒，停止执行")
                    break
                obs = deploy_env.get_observation()
                traj_rgb_np, robot_state = get_obs(obs, stats)
                image_list.append(traj_rgb_np)
                robot_state = torch.from_numpy(robot_state).to(dtype=model_dtype).cuda().unsqueeze(0)
                if t % query_frequency == 0:
                    curr_image = torch.from_numpy(traj_rgb_np / 255.0).to(dtype=model_dtype).cuda()
                if t == 0:
                    for _ in range(10):
                        batch = policy.process_batch_to_llava(curr_image, robot_state, raw_lang)
                        policy.policy(**batch, eval=True)
                    print('network warm up done')
                    time1 = time.time()
                # 定期重置模型状态，防止数值累积误差
                if t > 0 and t % 50 == 0:
                    print(f"步骤 {t}: 重置模型状态以保持数值稳定性")
                    # 重新设置模型为评估模式
                    policy.policy.eval()
                    # 清除可能的缓存
                    if hasattr(policy.policy, 'clear_cache'):
                        policy.policy.clear_cache()
                if policy_config['action_head_type'] in ["act", "droid_diffusion"]:
                    if t % query_frequency == 0:
                        batch = policy.process_batch_to_llava(curr_image, robot_state, raw_lang)
                        # 添加输入数据检查
                        for key, value in batch.items():
                            if isinstance(value, torch.Tensor):
                                if torch.any(torch.isnan(value)):
                                    print(f"警告: 输入数据 {key} 包含NaN值")
                                    # 使用安全值替换
                                    batch[key] = torch.nan_to_num(value, nan=0.0)
                        # 在关键步骤添加调试
                        if t % 20 == 0:  # 每20步调试一次
                            debug_model_outputs(policy, batch, t)
                        all_actions = policy.policy(**batch, eval=True)
                        # 检查模型输出是否包含NaN
                        if torch.any(torch.isnan(all_actions)):
                            print(f"警告: 模型输出包含NaN值，使用零动作")
                            all_actions = torch.zeros_like(all_actions)
                    if temporal_agg:
                        all_time_actions[[t], t:t + num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        # 检查是否有有效的动作
                        if len(actions_for_curr_step) == 0:
                            print(f"警告: 没有有效的动作，使用零动作")
                            raw_action = torch.zeros((1, action_dim), dtype=model_dtype).cuda()
                        else:
                            k = 0.01
                            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                            exp_weights = exp_weights / exp_weights.sum()
                            exp_weights = torch.from_numpy(exp_weights).cuda().to(dtype=actions_for_curr_step.dtype).unsqueeze(dim=1)
                            raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % int(query_frequency)]
                else:
                    raise NotImplementedError
                print(f"raw action size: {raw_action.size()}")
                # 检查原始动作是否包含NaN
                if torch.any(torch.isnan(raw_action)):
                    print(f"警告: 原始动作包含NaN值，使用零动作")
                    raw_action = torch.zeros_like(raw_action)
                raw_action = raw_action.squeeze(0).cpu().numpy()
                # 检查numpy数组是否包含NaN
                if np.any(np.isnan(raw_action)):
                    print(f"警告: 转换后的动作包含NaN值，使用零动作")
                    raw_action = np.zeros_like(raw_action)
                # 添加调试输出
                print(f"原始模型输出 (raw_action): {raw_action}")
                print(f"原始模型输出形状: {raw_action.shape}")
                action = post_process(raw_action)
                print(f"后处理后的动作 (post_process): {action}")
                print(f"after post_process action size: {action.shape}")
                # 根据任务类型传递参数给动作转换函数
                task_type = "pick_up_bowl" if "bowl" in raw_lang.lower() else "general"
                # 使用动作平滑，根据步数调整平滑因子
                # 早期步数使用较强平滑，后期减少平滑
                if t < 50:
                    smoothing_factor = 0.5  # 前50步使用较强平滑
                elif t < 100:
                    smoothing_factor = 0.3  # 50-100步使用中等平滑
                else:
                    smoothing_factor = 0.1  # 100步后使用较弱平滑

                # ✅ 获取当前末端执行器位置
                current_ee_pos = deploy_env.get_current_ee_position()
                print(f"步骤 {t}: 当前末端执行器位置: {current_ee_pos}")

                # 在转换动作之前，检查是否存在异常大的跳跃
                if last_action is not None:
                    # 检查位置变化是否过大
                    pos_change = np.linalg.norm(action[:3] - last_action[:3])
                    if pos_change > 0.2:  # 如果位置变化超过20cm，可能是异常值
                        print(f"警告: 检测到异常大的位置变化 ({pos_change:.3f}m)，应用限制")
                        # 限制位置变化
                        direction = action[:3] - last_action[:3]
                        direction = direction / np.linalg.norm(direction)
                        action[:3] = last_action[:3] + direction * 0.2

                # ✅ 传入当前末端执行器位置，将action解释为相对位移
                action = convert_actions(action, task_type=task_type,
                                        last_action=last_action,
                                        smoothing_factor=smoothing_factor,
                                        current_ee_pos=current_ee_pos)

                # 动作后处理：检查四元数是否有效
                if len(action) >= 7:
                    quat = action[3:7]
                    quat_norm = np.linalg.norm(quat)
                    if abs(quat_norm - 1.0) > 0.1:
                        print(f"警告: 四元数范数异常 ({quat_norm:.3f})，重新归一化")
                        if quat_norm > 0:
                            action[3:7] = quat / quat_norm
                        else:
                            action[3:7] = np.array([0.0, 0.0, 0.0, 1.0])  # 使用单位四元数

                # 夹爪控制平滑处理
                # 注意: 动作格式为 [x, y, z, rot_6d(6维), gripper(1维)] = 10维
                # 夹爪在索引9 (第10维)
                if len(action) >= 10 and last_action is not None and len(last_action) >= 10:
                    # 对夹爪控制进行平滑处理，避免突变
                    current_gripper = action[9]
                    last_gripper = last_action[9]
                    gripper_change = abs(current_gripper - last_gripper)

                    # 如果夹爪变化过大，进行限制
                    if gripper_change > 0.02:  # 2cm的变化限制
                        print(f"警告: 夹爪变化过大 ({gripper_change:.3f}m)，进行限制")
                        max_change = 0.02
                        if current_gripper > last_gripper:
                            action[9] = last_gripper + max_change
                        else:
                            action[9] = last_gripper - max_change
                        print(f"夹爪值从 {current_gripper:.3f} 调整为 {action[9]:.3f}")
                print(f'step {t}, 最终动作 (pred action): {action}')
                # 优化位置检测和终止条件
                current_position = action[:3]
                if last_position is not None:
                    position_change = np.linalg.norm(current_position - last_position)
                    # 只有当执行足够步数后才开始检测稳定性
                    if t > min_steps_for_completion:
                        if position_change < target_reached_threshold:
                            stationary_count += 1
                        else:
                            stationary_count = 0
                            success_count = 0
                        # 检查是否成功接近目标（基于动作稳定性而不是预设位置）
                        # 移除硬编码的目标位置检测，让模型自主决定何时完成任务
                        # 只基于动作稳定性来判断任务完成
                        if position_change < target_reached_threshold:
                            success_count += 1
                        else:
                            success_count = 0
                        # 改进的终止条件
                        # 需要同时满足：位置稳定 且 执行足够步数
                        if (stationary_count > 30 and
                            success_count > 20 and
                            t > min_steps_for_completion):
                            print(f"任务完成！机器人位置已稳定，停止执行")
                            print(f"最终位置: {current_position}")
                            break
                last_position = current_position.copy()
                last_action = action.copy()  # 更新上一个动作
                obs, done, info = deploy_env.step(action)

                # 添加额外的安全检查
                # 检查关节位置是否超出安全范围
                if hasattr(deploy_env, 'joint_positions') and len(deploy_env.joint_positions) >= 7:
                    if not deploy_env.check_joint_safety(deploy_env.joint_positions):
                        print("检测到关节超出安全范围，执行紧急停止!")
                        deploy_env.emergency_stop()
                        break  # 停止执行

                robot_state_list.append(robot_state)
                target_action_list.append(action)
                duration = time.time() - time1
                sleep_time = max(0, DT - duration)
                time.sleep(sleep_time)
                if duration >= DT:
                    culmulated_delay += (duration - DT)
                    print(f'Warning: step duration: {duration:.3f} s at step {t} longer than DT: {DT} s, culmulated delay: {culmulated_delay:.3f} s')
                # 添加键盘中断检查
                if rospy.is_shutdown():
                    print("ROS关闭信号接收，停止执行")
                    break
            print(f'Avg fps {max_timesteps / (time.time() - time0)}')
            plt.close()
    return
if __name__ == '__main__':
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    action_head = 'droid_diffusion'
    policy_config = {
        "model_path": "/home/tianxiaoyan/TinyVLA/output/droid_multi_task_processed_latest",
        "model_base": "./checkpoints/llava-pythia-13b",
        "enable_lora": True,
        "conv_mode": "pythia",
        "action_head": action_head,
        "action_head_type": action_head,
    }
    raw_lang = 'pick up the white bowl'
    try:
        print("正在加载策略模型...")
        policy = llava_pythia_act_policy(policy_config)
        print("策略模型加载完成")
        print("正在初始化机器人环境...")
        deploy_env = FrankaROSEnvironment(left_cam_id=4, right_cam_id=10)
        # 设置全局变量
        global_deploy_env = deploy_env
        print("机器人环境初始化完成")

        # 安全初始化：发送初始安全位置
        print("执行安全初始化...")
        deploy_env.send_safe_position()
        print("安全初始化完成")

        eval_bc(policy, deploy_env, policy_config, save_episode=True, num_rollouts=1, raw_lang=raw_lang)
    except KeyboardInterrupt:
        print("用户中断执行")
        # 执行紧急停止
        if global_deploy_env is not None:
            global_deploy_env.emergency_stop()
    except rospy.ROSInterruptException:
        print("ROS中断")
    except Exception as e:
        print(f"运行错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保ROS节点正确关闭
        if not rospy.is_shutdown():
            rospy.signal_shutdown("程序执行完成")
        print("程序已退出")