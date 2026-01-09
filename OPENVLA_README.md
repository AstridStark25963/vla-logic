# OpenVLA-7B 集成指南

本文档说明如何在现有的 VLA 控制系统中使用 OpenVLA-7B 模型。

## 概述

OpenVLA-7B 是一个开源的视觉-语言-动作模型，具有以下特点：

- **视觉编码器**: DINOv2 + SigLIP 双编码器
- **语言模型**: Llama-2-7b
- **图像尺寸**: 224×224 (不同于 TinyVLA 的 320×320)
- **动作输出**: 直接输出连续动作空间，无需扩散解码器

## 前置要求

### 1. 模型下载

OpenVLA-7B 模型应已下载到本地路径：
```
~/Desktop/openvla/openvla-7b
```

如果模型在其他位置，请修改 `openvla_config.py` 中的 `model_path`。

### 2. 依赖安装

确保已安装以下 Python 包：
```bash
pip install transformers torch torchvision numpy pillow
```

## 使用方法

### 方式 1: 使用配置文件

在 `inference_server.py` 中导入配置：

```python
from openvla_config import get_openvla_config

# 在 main() 函数中
policy_config = get_openvla_config()
```

### 方式 2: 直接修改配置

在 `inference_server.py` 的 `main()` 函数中，设置：

```python
USE_POLICY = 'openvla'  # 改为 'openvla' 以使用 OpenVLA-7B
```

默认已设置为 OpenVLA。

### 方式 3: 在代码中直接使用

```python
from eval_real_franka import openvla_act_policy

# 创建策略实例
policy_config = {
    "policy_type": "openvla",
    "model_path": "~/Desktop/openvla/openvla-7b",
    "action_dim": 7,
    "chunk_size": 50,
}

policy = openvla_act_policy(policy_config)
```

## 配置说明

### OpenVLA 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `policy_type` | str | "openvla" | 策略类型标识 |
| `model_path` | str | "~/Desktop/openvla/openvla-7b" | 模型路径 |
| `image_size` | int | 224 | 输入图像尺寸 |
| `action_dim` | int | 7 | 动作维度 (7 DOF) |
| `chunk_size` | int | 50 | 动作序列长度 |

### 任务描述

任务描述应使用自然语言，例如：
- "pick up the white bowl"
- "pick up the wooden block and place it in the blue basket"
- "grasp the cup"

## 架构对比

### TinyVLA vs OpenVLA

| 特性 | TinyVLA | OpenVLA-7B |
|------|---------|------------|
| 视觉编码器 | CLIP ViT | DINOv2 + SigLIP |
| 语言模型 | Pythia-1.3B | Llama-2-7b |
| 图像尺寸 | 320×320 | 224×224 |
| 动作解码 | Diffusion Policy | 直接输出 |
| 参数量 | ~1.3B | ~7B |

## 推理流程

### OpenVLA 推理步骤

1. **图像预处理**:
   - 调整到 224×224
   - ImageNet 归一化: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

2. **输入构建**:
   ```python
   inputs = processor(
       text=f"USER: What action should the robot take to {task}?\nASSISTANT:",
       images=image,
       return_tensors="pt"
   )
   ```

3. **动作预测**:
   ```python
   action = model.predict_action(**inputs, unnorm_key="bridge_orig")
   ```

4. **输出格式**: `(batch, horizon, action_dim)` 或 `(horizon, action_dim)`

## 双机部署

系统保持原有的双机部署架构：

### 主机 A (推理服务器)
- 运行 `inference_server.py`
- 负责模型推理
- 发布动作序列

### 主机 B (控制客户端)
- 运行 `control_client_fixed.py`
- 负责传感器数据采集
- 执行机器人控制

## ROS Topics

保持与原系统相同的 ROS topics：

- `/camera/left/image_raw`: 左相机图像
- `/camera/right/image_raw`: 右相机图像
- `/robot/state`: 机器人状态
- `/inference/actions`: 推理动作序列
- `/cartesian_impedance_example_controller/equilibrium_pose`: 目标位姿
- `/franka_gripper/move`: 夹爪控制

## 启动步骤

### 1. 启动 ROS Master (在主机 B)
```bash
roscore
```

### 2. 启动推理服务器 (在主机 A)
```bash
cd /path/to/vla-logic
python inference_server.py
```

### 3. 启动控制客户端 (在主机 B)
```bash
cd /path/to/vla-logic
python control_client_fixed.py
```

## 故障排除

### 问题 1: 模型路径找不到

**错误**: `FileNotFoundError: 模型路径不存在`

**解决方案**:
- 检查模型是否已下载到 `~/Desktop/openvla/openvla-7b`
- 或修改 `openvla_config.py` 中的 `model_path`

### 问题 2: CUDA 内存不足

**错误**: `RuntimeError: CUDA out of memory`

**解决方案**:
- OpenVLA-7B 需要约 14GB GPU 内存
- 确保 GPU 有足够的可用内存
- 关闭其他 GPU 进程

### 问题 3: 图像尺寸不匹配

**错误**: `RuntimeError: size mismatch`

**解决方案**:
- 确保输入图像正确调整到 224×224
- 检查 `process_batch_to_llava` 方法中的图像预处理

### 问题 4: 动作维度不匹配

**错误**: 动作输出维度与预期不符

**解决方案**:
- 检查 `action_dim` 设置是否为 7
- 确认 `chunk_size` 设置正确
- 验证数据集统计文件 (`dataset_stats.pkl`) 存在

## 性能优化

### 1. 推理加速
- 使用 `torch.float16` (半精度)
- 启用 `torch.compile()` (PyTorch 2.0+)
- 使用批处理推理

### 2. 内存优化
- 减小 `chunk_size`
- 使用 gradient checkpointing
- 清理 GPU 缓存: `torch.cuda.empty_cache()`

## 开发与调试

### 测试配置
```bash
python openvla_config.py
```

### 验证模型加载
```python
from eval_real_franka import openvla_act_policy
import torch

config = {
    "policy_type": "openvla",
    "model_path": "~/Desktop/openvla/openvla-7b",
    "action_dim": 7,
    "chunk_size": 50,
}

try:
    policy = openvla_act_policy(config)
    print("✅ 模型加载成功")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
```

### 日志记录

推理服务器和控制客户端都会生成日志文件：
- `inference_server_<timestamp>.log`
- `control_client_<timestamp>.log`

查看日志以诊断问题：
```bash
tail -f inference_server_*.log
```

## 参考资源

- [OpenVLA GitHub](https://github.com/openvla/openvla)
- [OpenVLA 论文](https://arxiv.org/abs/2406.09246)
- [Transformers 文档](https://huggingface.co/docs/transformers)

## 贡献

如有问题或改进建议，请提交 Issue 或 Pull Request。

## 许可证

本项目遵循原项目许可证。
