"""
OpenVLA-7B 配置文件

此文件包含 OpenVLA-7B 模型的配置参数
可以在 inference_server.py 和 eval_real_franka.py 中导入使用
"""

# OpenVLA-7B 配置
OPENVLA_CONFIG = {
    # 策略类型: 'tinyvla' 或 'openvla'
    "policy_type": "openvla",
    
    # 模型路径 (本地已下载的 OpenVLA-7B 模型)
    "model_path": "~/Desktop/openvla/openvla-7b",
    
    # 图像尺寸 (OpenVLA 使用 224x224，不同于 TinyVLA 的 320x320)
    "image_size": 224,
    
    # 机器人动作维度 (7 DOF 关节 + 1 夹爪)
    "action_dim": 7,
    
    # 动作序列长度 (预测的未来动作步数)
    "chunk_size": 50,
    
    # 设备
    "device": "cuda",
    
    # 数据类型
    "dtype": "torch.float32",
}

# TinyVLA 配置 (保持原有配置作为参考)
TINYVLA_CONFIG = {
    "policy_type": "tinyvla",
    "model_path": "/home/tianxiaoyan/TinyVLA/output/droid_multi_task_processed_latest",
    "model_base": "./checkpoints/llava-pythia-13b",
    "enable_lora": True,
    "conv_mode": "pythia",
    "action_head": "droid_diffusion",
    "action_head_type": "droid_diffusion",
}

# 任务描述配置
TASK_DESCRIPTIONS = {
    "pick_bowl": "pick up the white bowl",
    "pick_block": "pick up the wooden block and place it in the blue basket",
    "pick_cup": "pick up the cup",
    "place_object": "place the object on the table",
}

# OpenVLA 特定配置
OPENVLA_SPECS = {
    # 视觉编码器
    "vision_encoders": ["DINOv2", "SigLIP"],
    
    # 语言模型
    "language_model": "Llama-2-7b",
    
    # 图像归一化参数 (ImageNet 标准)
    "image_mean": [0.485, 0.456, 0.406],
    "image_std": [0.229, 0.224, 0.225],
    
    # 动作空间类型
    "action_space": "continuous",
    
    # 反归一化 key (用于 predict_action)
    "unnorm_key": "bridge_orig",
}

# Franka Panda 机器人配置
FRANKA_CONFIG = {
    # 关节限制 (弧度)
    "joint_limits": {
        "min": [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
        "max": [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
    },
    
    # 工作空间限制 (米)
    "workspace_limits": {
        "x": [0.1, 0.6],   # 前后
        "y": [-0.4, 0.4],  # 左右
        "z": [0.05, 0.7],  # 上下
    },
    
    # 夹爪参数
    "gripper": {
        "max_width": 0.08,  # 最大开合距离 (米)
        "min_width": 0.0,   # 最小开合距离 (米)
    }
}

# ROS 配置
ROS_CONFIG = {
    # Topic 名称
    "topics": {
        "camera_left": "/camera/left/image_raw",
        "camera_right": "/camera/right/image_raw",
        "robot_state": "/robot/state",
        "inference_actions": "/inference/actions",
        "equilibrium_pose": "/cartesian_impedance_example_controller/equilibrium_pose",
        "joint_states": "/franka_state_controller/joint_states",
        "gripper_move": "/franka_gripper/move",
    },
    
    # 控制频率 (Hz)
    "control_rate": 5,
    
    # 推理频率 (Hz)
    "inference_rate": 1,
    
    # 传感器发布频率 (Hz)
    "sensor_publish_rate": 10,
}


def get_openvla_config():
    """获取 OpenVLA 配置"""
    return OPENVLA_CONFIG.copy()


def get_tinyvla_config():
    """获取 TinyVLA 配置"""
    return TINYVLA_CONFIG.copy()


def get_config_by_type(policy_type="openvla"):
    """
    根据策略类型获取配置
    
    Args:
        policy_type: 'openvla' 或 'tinyvla'
    
    Returns:
        dict: 策略配置
    """
    if policy_type == "openvla":
        return get_openvla_config()
    elif policy_type == "tinyvla":
        return get_tinyvla_config()
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


if __name__ == "__main__":
    # 测试配置
    print("="*60)
    print("OpenVLA 配置:")
    print("="*60)
    for key, value in OPENVLA_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("TinyVLA 配置:")
    print("="*60)
    for key, value in TINYVLA_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("OpenVLA 特性:")
    print("="*60)
    for key, value in OPENVLA_SPECS.items():
        print(f"  {key}: {value}")
