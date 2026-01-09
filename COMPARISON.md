# TinyVLA vs OpenVLA-7B 实现对比

本文档详细对比两种 VLA 模型在本系统中的实现差异。

## 1. 模型架构对比

### TinyVLA (LLaVA-Pythia)

```
输入层:
  - 视觉编码器: CLIP ViT-L/14
  - 语言模型: Pythia-1.3B (GPT-NeoX)
  - 图像尺寸: 320×320

处理流程:
  1. 图像 → CLIP ViT → 视觉特征
  2. 文本指令 → Pythia tokenizer → 文本嵌入
  3. [视觉特征 + 文本嵌入] → Pythia decoder
  4. Pythia 输出 → Diffusion Policy → 动作序列

动作解码:
  - 使用扩散策略 (Diffusion Policy)
  - 需要多步迭代采样
  - 输出: (batch, chunk_size, action_dim)
```

### OpenVLA-7B

```
输入层:
  - 视觉编码器: DINOv2 (224×224) + SigLIP (224×224)
  - 语言模型: Llama-2-7b
  - 图像尺寸: 224×224 (双编码器)

处理流程:
  1. 图像 → DINOv2 + SigLIP → 融合视觉特征
  2. 文本指令 → Llama-2 tokenizer → 文本嵌入
  3. [融合特征 + 文本嵌入] → Llama-2 decoder
  4. Llama-2 输出 → 线性投影头 → 动作序列

动作解码:
  - 直接输出 (无扩散过程)
  - 单步前向传播
  - 输出: (batch, horizon, action_dim)
```

## 2. 代码实现对比

### 2.1 策略类初始化

#### TinyVLA
```python
from llava_pythia.model import LlavaPythiaForCausalLM
from peft import PeftModel

# 加载基础模型
base_model = LlavaPythiaForCausalLM.from_pretrained(
    model_base,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=False,
    device_map=None,
    trust_remote_code=True
).cpu()

# 加载 LoRA 权重
policy = PeftModel.from_pretrained(base_model, model_path)
policy = policy.merge_and_unload()
```

#### OpenVLA
```python
from transformers import AutoModelForVision2Seq, AutoProcessor

# 一步加载完整模型
policy = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).cuda()

# 加载处理器
processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True
)
```

### 2.2 图像预处理

#### TinyVLA (320×320 + CLIP 归一化)
```python
# 调整尺寸
img_resized = torch.nn.functional.interpolate(
    img.unsqueeze(0), 
    size=(320, 320), 
    mode='bilinear'
).squeeze(0)

# CLIP 归一化
mean = [0.48145466, 0.4578275, 0.40821073]
std = [0.26862954, 0.26130258, 0.27577711]
img_norm = (img_resized - mean) / std
```

#### OpenVLA (224×224 + ImageNet 归一化)
```python
# 调整尺寸
img_resized = torch.nn.functional.interpolate(
    img.unsqueeze(0), 
    size=(224, 224), 
    mode='bilinear'
).squeeze(0)

# ImageNet 归一化
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img_norm = (img_resized - mean) / std
```

### 2.3 推理流程

#### TinyVLA
```python
# 准备输入批次
batch = policy.process_batch_to_llava(curr_image, robot_state, task)

# 推理 (扩散采样)
with torch.inference_mode():
    all_actions = policy.policy(**batch, eval=True)

# 后处理
raw_actions = all_actions[0].cpu().numpy()
processed_actions = post_process(raw_actions)
```

#### OpenVLA
```python
# 准备输入
batch = policy.process_batch_to_llava(curr_image, robot_state, task)

# 使用 processor
inputs = policy.processor(
    text=batch['prompt'],
    images=batch['images'],
    return_tensors="pt"
).to("cuda")

# 推理 (直接预测)
with torch.inference_mode():
    actions = policy.policy.predict_action(
        **inputs, 
        unnorm_key="bridge_orig"
    )

# 后处理
processed_actions = post_process(actions.cpu().numpy())
```

## 3. 性能对比

| 指标 | TinyVLA | OpenVLA-7B |
|------|---------|------------|
| **模型大小** | ~1.3B 参数 | ~7B 参数 |
| **GPU 内存** | ~6GB | ~14GB |
| **推理速度** | ~100ms/step | ~150ms/step (预估) |
| **图像尺寸** | 320×320 | 224×224 |
| **推理步骤** | 多步 (扩散) | 单步 (直接) |
| **精度** | Float32 | Float32 |

## 4. 配置文件对比

### TinyVLA 配置
```python
{
    "policy_type": "tinyvla",
    "model_path": "/path/to/tinyvla/checkpoint",
    "model_base": "./checkpoints/llava-pythia-13b",
    "enable_lora": True,
    "conv_mode": "pythia",
    "action_head": "droid_diffusion",
    "action_head_type": "droid_diffusion",
}
```

### OpenVLA 配置
```python
{
    "policy_type": "openvla",
    "model_path": "~/Desktop/openvla/openvla-7b",
    "action_dim": 7,
    "chunk_size": 50,
    "image_size": 224,
}
```

## 5. 关键差异总结

### 5.1 视觉编码器
- **TinyVLA**: 单编码器 (CLIP ViT)
- **OpenVLA**: 双编码器 (DINOv2 + SigLIP)
  - DINOv2: 自监督学习，擅长物体检测和分割
  - SigLIP: 对比学习，擅长图文对齐

### 5.2 语言模型
- **TinyVLA**: Pythia-1.3B (更小、更快)
- **OpenVLA**: Llama-2-7b (更大、更强)

### 5.3 动作生成
- **TinyVLA**: 扩散策略 (多步采样)
  - 优点: 可以建模多模态分布
  - 缺点: 推理慢
- **OpenVLA**: 直接回归 (单步输出)
  - 优点: 推理快
  - 缺点: 单模态输出

### 5.4 训练数据
- **TinyVLA**: 可能使用较小数据集
- **OpenVLA**: 使用 Open X-Embodiment (跨具身化大规模数据集)

## 6. 使用建议

### 选择 TinyVLA 当:
- GPU 内存有限 (<8GB)
- 需要快速推理
- 任务相对简单
- 已有训练好的 checkpoint

### 选择 OpenVLA-7B 当:
- GPU 内存充足 (>=14GB)
- 需要更强的泛化能力
- 复杂的多步骤任务
- 想要开箱即用的预训练模型

## 7. 迁移指南

### 从 TinyVLA 迁移到 OpenVLA

1. **更新配置**:
   ```bash
   python switch_policy.py openvla
   ```

2. **调整图像尺寸**: 320×320 → 224×224 (自动处理)

3. **修改归一化参数**: CLIP → ImageNet (自动处理)

4. **检查 GPU 内存**: 确保 >=14GB

5. **测试**: 运行 `test_openvla.py`

### 从 OpenVLA 迁移到 TinyVLA

1. **更新配置**:
   ```bash
   python switch_policy.py tinyvla
   ```

2. **准备模型**: 确保 TinyVLA checkpoint 存在

3. **检查 LoRA 权重**: 确保 `non_lora_trainables.bin` 存在

4. **测试**: 启动推理服务器验证

## 8. 常见问题

### Q1: 两个模型可以同时使用吗?
A: 不可以。系统一次只能加载一个模型。使用 `switch_policy.py` 切换。

### Q2: 哪个模型效果更好?
A: 取决于任务。OpenVLA-7B 通常泛化能力更强，但 TinyVLA 在特定任务上可能更快。

### Q3: 如何自定义模型?
A: 继承 `openvla_act_policy` 或 `llava_pythia_act_policy`，重写关键方法。

### Q4: 动作输出格式相同吗?
A: 是的。两个模型输出都是 `(chunk_size, action_dim)` 形状的动作序列。

## 9. 参考资料

- [OpenVLA 论文](https://arxiv.org/abs/2406.09246)
- [TinyVLA (LLaVA) 论文](https://arxiv.org/abs/2304.08485)
- [Open X-Embodiment 数据集](https://robotics-transformer-x.github.io/)
- [Diffusion Policy 论文](https://arxiv.org/abs/2303.04137)
