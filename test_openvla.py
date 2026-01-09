#!/usr/bin/env python3
"""
OpenVLA-7B 测试脚本

用于验证 OpenVLA 模型是否正确加载和运行
不需要 ROS 或机器人硬件，仅测试模型本身
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_loading():
    """测试模型加载"""
    print("=" * 60)
    print("测试 1: 模型加载")
    print("=" * 60)
    
    try:
        from eval_real_franka import openvla_act_policy
        
        policy_config = {
            "policy_type": "openvla",
            "model_path": "~/Desktop/openvla/openvla-7b",
            "action_dim": 7,
            "chunk_size": 50,
        }
        
        print("正在加载 OpenVLA 模型...")
        policy = openvla_act_policy(policy_config)
        
        print("✅ 模型加载成功")
        print(f"   设备: {next(policy.policy.parameters()).device}")
        print(f"   数据类型: {next(policy.policy.parameters()).dtype}")
        print(f"   动作维度: {policy.config.action_dim}")
        print(f"   序列长度: {policy.config.chunk_size}")
        
        return policy
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_image_processing(policy):
    """测试图像处理"""
    print("\n" + "=" * 60)
    print("测试 2: 图像处理")
    print("=" * 60)
    
    try:
        # 创建模拟图像 (2 个相机，640x480, RGB)
        left_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        right_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        print(f"创建模拟图像:")
        print(f"  左相机: {left_image.shape}")
        print(f"  右相机: {right_image.shape}")
        
        # 转换为 torch tensor (归一化到 [0, 1])
        images = np.stack([left_image, right_image], axis=0)
        curr_image = torch.from_numpy(images / 255.0).float().cuda()
        
        print(f"转换为 tensor: {curr_image.shape}")
        
        # 创建模拟机器人状态 (7 DOF)
        robot_state = torch.randn(1, 7).float().cuda()
        print(f"机器人状态: {robot_state.shape}")
        
        # 任务描述
        task_description = "pick up the white bowl"
        print(f"任务描述: {task_description}")
        
        # 处理输入
        print("\n正在处理输入批次...")
        batch = policy.process_batch_to_llava(
            curr_image, robot_state, task_description
        )
        
        print("✅ 图像处理成功")
        print(f"   批次 keys: {batch.keys()}")
        print(f"   图像形状: {batch['images'].shape}")
        print(f"   提示词: {batch['prompt'][:80]}...")
        print(f"   状态形状: {batch['states'].shape}")
        
        return batch
        
    except Exception as e:
        print(f"❌ 图像处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_inference(policy, batch):
    """测试推理"""
    print("\n" + "=" * 60)
    print("测试 3: 模型推理")
    print("=" * 60)
    
    try:
        print("正在运行推理...")
        
        with torch.inference_mode():
            # 使用 processor 处理输入
            # 将图像转换回 numpy 格式 (HWC)
            image_np = batch['images'].cpu().numpy().transpose(0, 2, 3, 1)
            
            inputs = policy.processor(
                text=batch['prompt'],
                images=image_np,
                return_tensors="pt"
            ).to("cuda")
            
            print(f"Processor 输入 keys: {inputs.keys()}")
            
            # 检查模型是否有 predict_action 方法
            if hasattr(policy.policy, 'predict_action'):
                print("使用 predict_action 方法...")
                actions = policy.policy.predict_action(
                    **inputs, 
                    unnorm_key="bridge_orig"
                )
            else:
                print("使用标准 forward 方法...")
                actions = policy.policy(**inputs)
        
        print("✅ 推理成功")
        print(f"   输出形状: {actions.shape}")
        print(f"   输出范围: [{actions.min():.3f}, {actions.max():.3f}]")
        print(f"   输出样本 (前 5 个值): {actions.flatten()[:5].cpu().numpy()}")
        
        # 验证输出维度
        if actions.dim() == 3:
            batch_size, horizon, action_dim = actions.shape
            print(f"   批次大小: {batch_size}")
            print(f"   时间步长: {horizon}")
            print(f"   动作维度: {action_dim}")
        elif actions.dim() == 2:
            horizon, action_dim = actions.shape
            print(f"   时间步长: {horizon}")
            print(f"   动作维度: {action_dim}")
        
        return actions
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_config():
    """测试配置文件"""
    print("\n" + "=" * 60)
    print("测试 4: 配置文件")
    print("=" * 60)
    
    try:
        from openvla_config import get_openvla_config, OPENVLA_SPECS
        
        config = get_openvla_config()
        
        print("✅ 配置加载成功")
        print("\nOpenVLA 配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        print("\nOpenVLA 特性:")
        for key, value in OPENVLA_SPECS.items():
            print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("OpenVLA-7B 测试套件")
    print("=" * 60)
    print()
    
    # 检查 CUDA 是否可用
    if not torch.cuda.is_available():
        print("⚠️  警告: CUDA 不可用，测试可能失败")
        print(f"   PyTorch 版本: {torch.__version__}")
        return
    
    print(f"✅ CUDA 可用")
    print(f"   设备数量: {torch.cuda.device_count()}")
    print(f"   当前设备: {torch.cuda.current_device()}")
    print(f"   设备名称: {torch.cuda.get_device_name()}")
    print(f"   显存总量: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()
    
    # 运行测试
    results = {
        "model_loading": False,
        "image_processing": False,
        "inference": False,
        "config": False,
    }
    
    # 测试 1: 模型加载
    policy = test_model_loading()
    if policy is not None:
        results["model_loading"] = True
        
        # 测试 2: 图像处理
        batch = test_image_processing(policy)
        if batch is not None:
            results["image_processing"] = True
            
            # 测试 3: 推理
            actions = test_inference(policy, batch)
            if actions is not None:
                results["inference"] = True
    
    # 测试 4: 配置文件
    if test_config():
        results["config"] = True
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    print()
    if all_passed:
        print("🎉 所有测试通过！OpenVLA-7B 已正确配置。")
    else:
        print("⚠️  部分测试失败，请检查错误信息。")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
