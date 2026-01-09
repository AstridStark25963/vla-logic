# VLA Logic - Vision-Language-Action Control System

基于视觉-语言-动作(VLA)模型的机器人控制系统，支持 TinyVLA 和 OpenVLA-7B 模型。

## 功能特点

- ✅ **双模型支持**: 支持 TinyVLA (LLaVA-Pythia) 和 OpenVLA-7B
- ✅ **ROS 集成**: 完整的 ROS 通信机制
- ✅ **双目视觉**: 支持双相机输入处理
- ✅ **Franka 控制**: 适配 Franka Panda 机械臂 (7 DOF + 夹爪)
- ✅ **双机部署**: 推理服务器和控制客户端分离架构

## 项目结构

```
vla-logic/
├── eval_real_franka.py          # 策略类定义 (TinyVLA + OpenVLA)
├── inference_server.py          # 推理服务器 (主机 A)
├── control_client_fixed.py      # 控制客户端 (主机 B)
├── action_utils_fixed.py        # 动作转换工具
├── openvla_config.py            # OpenVLA 配置文件
├── test_openvla.py              # OpenVLA 测试脚本
├── OPENVLA_README.md            # OpenVLA 详细文档
└── README.md                    # 本文件
```

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install torch torchvision transformers numpy opencv-python rospy

# 确保 ROS 环境已配置
source /opt/ros/noetic/setup.bash  # 根据你的 ROS 版本
```

### 2. 模型选择

#### 使用 OpenVLA-7B (推荐)

1. 下载 OpenVLA-7B 模型到 `~/Desktop/openvla/openvla-7b`
2. 在 `inference_server.py` 中设置：
   ```python
   USE_POLICY = 'openvla'
   ```

#### 使用 TinyVLA

1. 确保 TinyVLA 模型在指定路径
2. 在 `inference_server.py` 中设置：
   ```python
   USE_POLICY = 'tinyvla'
   ```

### 3. 测试 OpenVLA (可选)

```bash
python test_openvla.py
```

### 4. 启动系统

#### 主机 B (控制客户端)
```bash
# 启动 ROS Master
roscore

# 在另一个终端启动控制客户端
python control_client_fixed.py
```

#### 主机 A (推理服务器)
```bash
python inference_server.py
```

## 模型对比

| 特性 | TinyVLA | OpenVLA-7B |
|------|---------|------------|
| 视觉编码器 | CLIP ViT | DINOv2 + SigLIP |
| 语言模型 | Pythia-1.3B | Llama-2-7b |
| 图像尺寸 | 320×320 | 224×224 |
| 动作解码 | Diffusion | 直接输出 |
| 参数量 | ~1.3B | ~7B |

## 配置说明

### OpenVLA 配置

编辑 `openvla_config.py` 或在代码中设置：

```python
OPENVLA_CONFIG = {
    "policy_type": "openvla",
    "model_path": "~/Desktop/openvla/openvla-7b",
    "image_size": 224,
    "action_dim": 7,
    "chunk_size": 50,
}
```

### 任务描述

支持自然语言任务描述，例如：
- "pick up the white bowl"
- "pick up the wooden block and place it in the blue basket"
- "grasp the cup"

## ROS Topics

- `/camera/left/image_raw`: 左相机图像
- `/camera/right/image_raw`: 右相机图像
- `/robot/state`: 机器人关节状态
- `/inference/actions`: 推理动作序列
- `/cartesian_impedance_example_controller/equilibrium_pose`: 笛卡尔目标位姿
- `/franka_gripper/move`: 夹爪控制 (Action)

## 文档

- [OpenVLA 详细文档](OPENVLA_README.md): OpenVLA-7B 集成指南
- [OpenVLA GitHub](https://github.com/openvla/openvla): 官方仓库

## 故障排除

### 常见问题

1. **模型加载失败**: 检查模型路径是否正确
2. **CUDA 内存不足**: OpenVLA-7B 需要约 14GB GPU 内存
3. **ROS 连接问题**: 确保 ROS Master 正确配置

详见 [OPENVLA_README.md](OPENVLA_README.md) 的故障排除章节。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

本项目遵循原项目许可证。

