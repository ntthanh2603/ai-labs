# 📚 Hướng Dẫn Toàn Diện về Fine-tuning Large Language Models (LLM)

## 📑 Mục Lục

1. [Tổng Quan](#tổng-quan)
2. [Các Phương Pháp Fine-tuning](#các-phương-pháp-fine-tuning)
3. [Parameter-Efficient Fine-Tuning (PEFT)](#parameter-efficient-fine-tuning-peft)
4. [Quantization - Giảm Memory](#quantization---giảm-memory)
5. [Hyperparameters Chi Tiết](#hyperparameters-chi-tiết)
6. [Dataset Preparation](#dataset-preparation)
7. [Training Strategies](#training-strategies)
8. [Monitoring & Debugging](#monitoring--debugging)
9. [Best Practices](#best-practices)
10. [Common Issues & Solutions](#common-issues--solutions)

---

## 🎯 Tổng Quan

### Fine-tuning là gì?

Fine-tuning là quá trình lấy một pre-trained model (đã học từ large corpus) và train thêm trên specific task/domain data để:

- Adapt model cho use case cụ thể
- Improve performance trên specialized tasks
- Inject domain knowledge
- Change behavior/style của model

### Khi nào cần Fine-tuning?

✅ **NÊN Fine-tune khi:**

- Model không có knowledge về domain cụ thể
- Cần output style/format đặc biệt
- Cần improve accuracy trên specific task
- Có dataset quality cao và relevant

❌ **KHÔNG NÊN Fine-tune khi:**

- Chỉ cần thay đổi prompts (thử prompt engineering trước)
- Dataset quá nhỏ (<100 samples)
- Dataset quality thấp hoặc noisy
- Chỉ cần thêm context (dùng RAG thay vì fine-tune)

---

## 🔧 Các Phương Pháp Fine-tuning

### 1. Full Fine-tuning (Traditional)

**Mô tả:** Train lại TẤT CẢ parameters của model

**Ưu điểm:**

- Maximum flexibility và adaptation
- Có thể thay đổi model hoàn toàn
- Best performance potential

**Nhược điểm:**

- Cực kỳ tốn memory (cần GPU với VRAM lớn)
- Training time lâu
- Risk catastrophic forgetting (quên kiến thức cũ)
- Expensive ($$$)

**Khi nào dùng:**

- Có resources mạnh (multiple A100/H100 GPUs)
- Dataset cực lớn (>100K samples)
- Cần completely retrain model

**Memory Requirements:**

```
7B model:  ~28GB (for weights only)
13B model: ~52GB
70B model: ~280GB

Trong training cần thêm:
- Optimizer states: 2x model size (Adam)
- Gradients: 1x model size
- Activations: depends on batch size

Total: ~4-8x model size
```

### 2. Parameter-Efficient Fine-Tuning (PEFT)

**Mô tả:** Freeze base model, chỉ train một phần nhỏ parameters

**Ưu điểm:**

- Tiết kiệm memory đáng kể (1-10% của full fine-tuning)
- Training nhanh hơn nhiều
- Có thể train trên consumer GPUs
- Ít risk overfitting
- Easy to manage multiple adapters

**Nhược điểm:**

- Giới hạn adaptation capacity
- Có thể không đủ cho dramatic changes

**Các kỹ thuật PEFT phổ biến:**

- LoRA (Low-Rank Adaptation) ⭐ Most popular
- QLoRA (Quantized LoRA)
- Prefix Tuning
- Adapter Layers
- Prompt Tuning

---

## 🎨 Parameter-Efficient Fine-Tuning (PEFT)

### LoRA (Low-Rank Adaptation)

**Concept cơ bản:**

Instead of updating weight matrix W:

```
W_new = W_old + ΔW
```

LoRA decomposes ΔW into two smaller matrices:

```
W_new = W_old + B × A
where:
- B: matrix [d × r]
- A: matrix [r × k]
- r: rank (r << min(d, k))
```

**Tại sao hiệu quả?**

- Original matrix: d × k parameters
- LoRA: (d × r) + (r × k) parameters
- Example:
  - Original: 4096 × 4096 = 16,777,216 params
  - LoRA (r=16): (4096×16) + (16×4096) = 131,072 params
  - Reduction: 128x fewer parameters!

### LoRA Parameters Explained

#### 1. **Rank (r)**

```python
r = 8    # Low rank - fewer params, less capacity
r = 16   # Sweet spot cho most tasks ⭐
r = 32   # Higher capacity
r = 64   # Very high capacity
r = 128  # Approaching full fine-tuning
```

**Ảnh hưởng:**

- **r thấp (4-8):**

  - ✅ Very memory efficient
  - ✅ Fast training
  - ❌ Limited adaptation capacity
  - 📊 Best for: Style changes, simple tasks

- **r trung bình (16-32):** ⭐

  - ✅ Good balance
  - ✅ Reasonable memory usage
  - ✅ Good performance
  - 📊 Best for: Most use cases

- **r cao (64-128):**
  - ✅ Maximum adaptation
  - ❌ More memory
  - ❌ Slower training
  - ⚠️ Risk overfitting
  - 📊 Best for: Complex tasks, large datasets

**Công thức tính trainable parameters:**

```python
trainable_params = 2 × r × d_model × num_target_modules

Example with Gemma-2-9B:
- d_model = 3584
- target_modules = 7 (q,k,v,o,gate,up,down)
- r = 16

trainable_params = 2 × 16 × 3584 × 7
                 = 802,816 parameters
                 = ~0.009% of 9B params
```

#### 2. **LoRA Alpha**

```python
lora_alpha = 16    # Common: same as r
lora_alpha = 32    # Common: 2x of r
lora_alpha = 8     # Lower scaling
```

**Công thức scaling:**

```python
scaling = lora_alpha / r

# Actual update applied:
ΔW = (lora_alpha / r) × B × A
```

**Ảnh hưởng:**

- **alpha = r:** Standard scaling (1.0x)
- **alpha = 2r:** Double the impact of LoRA
- **alpha < r:** Reduce LoRA influence
- **alpha > 2r:** Strong LoRA influence (risk instability)

**Best practices:**

- Start with `alpha = r`
- Increase if LoRA impact quá yếu
- Decrease if training unstable

#### 3. **Target Modules**

```python
# Minimal (memory efficient):
target_modules = ["q_proj", "v_proj"]

# Standard (recommended): ⭐
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Comprehensive:
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# All including embeddings:
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj",
                  "embed_tokens", "lm_head"]
```

**Module roles:**

- **q_proj, k_proj, v_proj:** Query, Key, Value trong Attention
  - Core của attention mechanism
  - Ảnh hưởng lớn đến context understanding
- **o_proj:** Output projection của Attention
  - Combine attention outputs
- **gate_proj, up_proj, down_proj:** MLP/FFN layers
  - Transform representations
  - Store factual knowledge
- **embed_tokens:** Input embeddings
  - Rarely needed unless new vocabulary
- **lm_head:** Output projection layer
  - Rarely needed unless changing output space

**Trade-offs:**
| Modules | Params | Memory | Quality | Use Case |
|---------|--------|--------|---------|----------|
| 2 (q,v) | Lowest | ~1GB | Basic | Style/format only |
| 4 (qkvo) | Low | ~2GB | Good | Most tasks ⭐ |
| 7 (all) | Medium | ~3GB | Better | Complex tasks |
| 9 (+ embed) | High | ~4GB | Best | New domain/vocab |

#### 4. **LoRA Dropout**

```python
lora_dropout = 0.0    # No dropout (fastest, Unsloth optimized) ⭐
lora_dropout = 0.05   # Light regularization
lora_dropout = 0.1    # Standard regularization
lora_dropout = 0.2    # Heavy regularization
```

**Khi nào dùng:**

- **0.0:**
  - Small to medium datasets
  - Want maximum speed
  - Using other regularization (weight_decay)
- **0.05-0.1:**
  - Dataset có signs of overfitting
  - Very small dataset (<500 samples)
- **0.2:**
  - Extremely small dataset
  - High overfitting risk

#### 5. **Bias Training**

```python
bias = "none"        # Don't train biases (fastest) ⭐
bias = "lora_only"   # Train only LoRA biases
bias = "all"         # Train all biases
```

**Recommendations:**

- `"none"`: Default choice, works well 99% of time
- `"all"`: Only if you notice bias issues

### QLoRA (Quantized LoRA)

Combination of Quantization + LoRA:

```python
# Load model in 4-bit
model = FastLanguageModel.from_pretrained(
    model_name="model-name",
    load_in_4bit=True,  # Quantize base model
)

# Add LoRA on top
model = get_peft_model(model, lora_config)
```

**Benefits:**

- 4-bit base model: ~4x memory reduction
- LoRA adapters in full precision
- Can train 65B models on single 24GB GPU!

**Memory savings:**

```
70B model full fine-tuning: ~280GB
70B model QLoRA: ~48GB
Reduction: ~6x
```

---

## 🔢 Quantization - Giảm Memory

### Quantization là gì?

Reduce precision of weights từ 32-bit float → lower bit integers.

### Quantization Methods

#### 1. **Float32 (Full Precision)**

```python
dtype = torch.float32
# 32 bits per parameter
# 7B model: 7B × 4 bytes = 28GB
```

- **Use case:** Research, maximum accuracy needed
- **Pros:** No precision loss
- **Cons:** Maximum memory

#### 2. **Float16 (Half Precision)**

```python
dtype = torch.float16
# 16 bits per parameter
# 7B model: 7B × 2 bytes = 14GB
```

- **Use case:** Training on older GPUs (T4, V100)
- **Pros:** 2x memory reduction, widely supported
- **Cons:** Risk of numerical instability, underflow

#### 3. **BFloat16**

```python
dtype = torch.bfloat16
# 16 bits but different format
# Better dynamic range than float16
```

- **Use case:** Modern GPUs (A100, H100, RTX 30/40 series)
- **Pros:** Better stability than float16, same memory savings
- **Cons:** Requires hardware support

**Float16 vs BFloat16:**

```
Float16:  1 sign | 5 exponent | 10 mantissa
BFloat16: 1 sign | 8 exponent | 7 mantissa
          └─ Same as Float32's exponent

BFloat16 has same range as Float32 but less precision
→ Better for deep learning (range > precision)
```

#### 4. **8-bit Quantization**

```python
load_in_8bit = True
# 7B model: ~7GB
```

- **Use case:** Inference, some fine-tuning
- **Pros:** 4x memory reduction
- **Cons:** Some accuracy loss (~1-2%)

#### 5. **4-bit Quantization** ⭐

```python
load_in_4bit = True
# 7B model: ~3.5GB
```

- **Use case:** Fine-tuning large models on consumer GPUs
- **Pros:** 8x memory reduction!
- **Cons:** ~2-5% accuracy loss (often acceptable)

**4-bit Quantization Types:**

```python
# NF4 (Normal Float 4) - Recommended ⭐
bnb_4bit_quant_type = "nf4"
# Optimized for normal distribution of weights
# Better quality than regular 4-bit

# FP4 (Float Point 4)
bnb_4bit_quant_type = "fp4"
# Standard 4-bit floating point
```

**Double Quantization:**

```python
bnb_4bit_use_double_quant = True
# Quantize the quantization constants themselves
# Extra ~0.4GB saving for 7B model
```

### Memory Comparison Table

| Model | FP32  | FP16/BF16 | 8-bit | 4-bit |
| ----- | ----- | --------- | ----- | ----- |
| 7B    | 28GB  | 14GB      | 7GB   | 3.5GB |
| 13B   | 52GB  | 26GB      | 13GB  | 6.5GB |
| 34B   | 136GB | 68GB      | 34GB  | 17GB  |
| 70B   | 280GB | 140GB     | 70GB  | 35GB  |

### Choosing Quantization

```python
# GPU < 12GB → Must use 4-bit
# GPU 12-24GB → Can use 8-bit or 4-bit
# GPU 24-48GB → Can use FP16/BF16 for smaller models
# GPU > 48GB → Can use full precision

# Example decision tree:
if gpu_memory < 12:
    load_in_4bit = True
elif gpu_memory < 24:
    load_in_8bit = True  # or 4-bit for larger models
else:
    dtype = torch.bfloat16  # or float16
```

---

## ⚙️ Hyperparameters Chi Tiết

### 1. Learning Rate

**Vai trò:** Kiểm soát tốc độ cập nhật weights

```python
learning_rate = 5e-5   # Very conservative
learning_rate = 1e-4   # Conservative
learning_rate = 2e-4   # Standard for LoRA ⭐
learning_rate = 5e-4   # Aggressive
learning_rate = 1e-3   # Very aggressive (risky)
```

**Finding the right learning rate:**

1. **Learning Rate Finder:**

```python
from transformers import Trainer

# Run LR finder
trainer.train()  # Will automatically find optimal LR
```

2. **Manual tuning signs:**

- **LR quá cao:**
  - Loss spikes/oscillates
  - Training diverges (loss → infinity)
  - NaN in gradients
- **LR quá thấp:**
  - Loss giảm rất chậm
  - Plateau sớm
  - Không converge trong reasonable time

3. **Best practices:**

```python
# LoRA fine-tuning
learning_rate = 2e-4  # Good default

# Full fine-tuning
learning_rate = 1e-5  # Much lower needed

# Smaller models (< 3B)
learning_rate = 3e-4  # Can be slightly higher

# Larger models (> 30B)
learning_rate = 1e-4  # Should be lower
```

### 2. Batch Size

**Batch size là số samples processed cùng lúc**

```python
per_device_train_batch_size = 1   # Minimum
per_device_train_batch_size = 2   # Small GPU
per_device_train_batch_size = 4   # Medium GPU ⭐
per_device_train_batch_size = 8   # Large GPU
per_device_train_batch_size = 16  # Very large GPU
```

**Gradient Accumulation:**

```python
# Simulate larger batch without more memory
gradient_accumulation_steps = 4

# Effective batch size:
effective_batch = per_device_batch × accumulation × num_gpus

Example:
- per_device_batch = 2
- accumulation = 4
- num_gpus = 1
→ effective_batch = 8
```

**Trade-offs:**

| Batch Size   | Memory | Speed | Stability | Generalization |
| ------------ | ------ | ----- | --------- | -------------- |
| Small (1-2)  | Low    | Slow  | Noisy     | Better         |
| Medium (4-8) | Medium | Good  | Good      | Good ⭐        |
| Large (16+)  | High   | Fast  | Stable    | May overfit    |

**Best practices:**

```python
# Limited memory: Use gradient accumulation
per_device_train_batch_size = 1
gradient_accumulation_steps = 8
# Effective batch = 8

# Enough memory: Maximize batch size
per_device_train_batch_size = 8
gradient_accumulation_steps = 1
# Both achieve same effective batch size
```

### 3. Training Steps/Epochs

```python
# Method 1: Fixed number of steps
max_steps = 500
num_train_epochs = -1  # Ignored

# Method 2: Number of epochs
num_train_epochs = 3
max_steps = -1  # Ignored

# Relationship:
steps_per_epoch = len(dataset) / (batch_size × accumulation_steps)
total_steps = steps_per_epoch × num_epochs
```

**How many steps/epochs?**

```python
# Dataset size guide:
if dataset_size < 100:
    epochs = 10-20  # More epochs needed
elif dataset_size < 500:
    epochs = 5-10
elif dataset_size < 1000:
    epochs = 3-5  ⭐
elif dataset_size < 5000:
    epochs = 2-3
else:
    epochs = 1-2

# Or use max_steps:
max_steps = dataset_size × 0.5 to 2.0
```

**Signs of right amount:**

- Loss converged (plateau)
- Validation metrics stop improving
- No signs of overfitting

**Signs of too much:**

- Validation loss increases while train loss decreases
- Model memorizing instead of learning
- Overfitting symptoms

### 4. Warmup Steps

**Vai trò:** Gradually increase LR from 0 to max để avoid instability

```python
warmup_steps = 0        # No warmup
warmup_steps = 10       # Short warmup
warmup_steps = 100      # Standard ⭐
warmup_steps = 500      # Long warmup

# Or as ratio:
warmup_ratio = 0.03     # 3% of total steps
warmup_ratio = 0.1      # 10% of total steps
```

**Warmup schedule visualization:**

```
LR
│     ┌─────────────────────  (max LR)
│    ╱
│   ╱
│  ╱  warmup phase
│ ╱
│╱
└─────────────────────────────> Steps
0   warmup_steps
```

**Best practices:**

```python
# Short training (< 500 steps)
warmup_steps = 10-50

# Medium training (500-2000 steps)
warmup_steps = 50-100  ⭐

# Long training (> 2000 steps)
warmup_steps = 100-500

# Or use ratio:
warmup_ratio = 0.05  # 5% of total steps
```

### 5. Weight Decay

**Vai trò:** L2 regularization để prevent overfitting

```python
weight_decay = 0.0    # No regularization
weight_decay = 0.01   # Light regularization ⭐
weight_decay = 0.1    # Standard regularization
weight_decay = 0.3    # Heavy regularization
```

**Formula:**

```python
loss = task_loss + weight_decay × Σ(weights²)
```

**When to use:**

- **0.01:** Default, works well for most cases
- **0.0:** If already using dropout or other regularization
- **0.1-0.3:** Small dataset with overfitting

### 6. Learning Rate Scheduler

**Vai trò:** Change LR during training

```python
# Linear decay (recommended for most cases) ⭐
lr_scheduler_type = "linear"
# LR decreases linearly to 0

# Cosine
lr_scheduler_type = "cosine"
# LR follows cosine curve

# Cosine with restarts
lr_scheduler_type = "cosine_with_restarts"
# Multiple cosine cycles

# Constant
lr_scheduler_type = "constant"
# LR stays same (use with warmup)

# Constant with warmup
lr_scheduler_type = "constant_with_warmup"
# Warmup then constant
```

**Visualization:**

```
Linear:
LR │\
   │ \
   │  \
   │   \________
   └────────────> Steps

Cosine:
LR │\
   │ ╲
   │  ╲
   │   ╲___
   │       ╲___
   └────────────> Steps

Cosine with Restarts:
LR │╲   ╲   ╲
   │ ╲   ╲   ╲
   │  ╲   ╲   ╲
   └────────────> Steps
```

**Best practices:**

- **Linear:** Most reliable, use by default
- **Cosine:** Slightly better for some tasks
- **Constant:** Only with very small LR

### 7. Optimizer

```python
# AdamW 8-bit (recommended) ⭐
optim = "adamw_8bit"
# Adam optimizer with 8-bit quantization
# Memory efficient

# Standard AdamW
optim = "adamw_torch"
# Full precision Adam
# Best quality but more memory

# SGD
optim = "sgd"
# Stochastic Gradient Descent
# Rarely used for LLMs

# Adafactor
optim = "adafactor"
# Memory efficient alternative
# Good for very large models
```

**Optimizer comparison:**

| Optimizer   | Memory   | Quality | Speed   | Use Case            |
| ----------- | -------- | ------- | ------- | ------------------- |
| AdamW 8-bit | Low      | Good    | Fast    | Default choice ⭐   |
| AdamW       | High     | Best    | Fast    | If memory available |
| SGD         | Lowest   | OK      | Fastest | Not recommended     |
| Adafactor   | Very Low | Good    | Medium  | Huge models         |

**Optimizer memory usage:**

```python
# For Adam/AdamW:
optimizer_memory = 2 × model_parameters × 4 bytes

Example 7B model:
= 2 × 7B × 4 bytes
= 56GB just for optimizer states!

# With 8-bit Adam:
= 2 × 7B × 1 byte
= 14GB
Saving: 42GB!
```

### 8. Gradient Clipping

```python
max_grad_norm = 1.0   # Default ⭐
max_grad_norm = 0.5   # More conservative
max_grad_norm = 2.0   # More lenient
max_grad_norm = None  # No clipping
```

**Vai trò:** Prevent exploding gradients

```python
# How it works:
if gradient_norm > max_grad_norm:
    gradient = gradient × (max_grad_norm / gradient_norm)
```

**When needed:**

- Training becomes unstable
- Loss spikes
- Getting NaN in gradients

### 9. Seed

```python
seed = 42      # Any number
seed = 3407    # Often used in papers
seed = None    # Random seed each run
```

**Importance:**

- Reproducibility
- Debugging
- Comparing experiments

**Best practice:**

```python
# Development: Use fixed seed
seed = 42

# Production: May use random seed
seed = None

# Research: Document seed used
seed = 3407  # Report in paper
```

---

## 📊 Dataset Preparation

### Dataset Size Guidelines

```python
# Minimum viable:
samples = 50-100
# Can work but high risk overfitting

# Small:
samples = 100-500
# Need careful regularization

# Medium:
samples = 500-2000  ⭐
# Good for most tasks

# Large:
samples = 2000-10000
# Excellent results possible

# Very large:
samples = 10000+
# Can achieve near state-of-art
```

### Dataset Quality > Quantity

**Quality checklist:**

- ✅ Accurate and correct information
- ✅ Consistent formatting
- ✅ Representative of target use case
- ✅ Diverse examples
- ✅ No duplicates
- ✅ Clean (no typos, formatting issues)

**One high-quality sample > Ten low-quality samples**

### Data Format

**Standard format (Alpaca):**

```json
{
  "instruction": "Question or task",
  "input": "Additional context (optional)",
  "output": "Expected response"
}
```

**Examples:**

```json
// Simple Q&A:
{
  "instruction": "What is the capital of France?",
  "input": "",
  "output": "The capital of France is Paris."
}

// With context:
{
  "instruction": "Summarize this text",
  "input": "Long text here...",
  "output": "Summary here..."
}

// Conversation:
{
  "instruction": "Continue this conversation",
  "input": "User: Hi, how are you?\nAssistant: I'm doing well, thank you!",
  "output": "How can I help you today?"
}
```

### Prompt Template Design

**Good template characteristics:**

1. Clear structure
2. Consistent formatting
3. Special tokens properly placed
4. Context separation

**Example template:**

```python
template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""
```

**Vietnamese template:**

```python
template = """Dưới đây là một câu hỏi về {topic}. Hãy trả lời chính xác và chi tiết.

### Câu hỏi:
{instruction}

### Ngữ cảnh:
{input}

### Trả lời:
{output}"""
```

**CRITICAL:** Always add EOS token!

```python
text = template.format(...) + tokenizer.eos_token
```

### Data Augmentation

**Techniques to increase data:**

1. **Paraphrasing:**

```python
# Original:
"What is your name?"

# Augmented:
"Can you tell me your name?"
"What do people call you?"
"How should I address you?"
```

2. **Back-translation:**

```
English → Vietnamese → English
(slightly different phrasing)
```

3. **Synthetic data generation:**

```python
# Use GPT-4/Claude to generate more examples
prompt = """
Generate 10 similar question-answer pairs about {topic}
Format: JSON with instruction and output
"""
```

4. **Template variations:**

```python
templates = [
    "Question: {q}\nAnswer: {a}",
    "Q: {q}\nA: {a}",
    "{q}\n\n{a}"
]
```

### Train/Validation Split

```python
# Standard split:
train_size = 0.9      # 90% train
validation_size = 0.1  # 10% validation

# For small datasets:
train_size = 0.8      # 80% train
validation_size = 0.2  # 20% validation

# For very small datasets (< 200):
# Use all for training, monitor train loss only
```

**K-fold cross-validation:**

```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(kfold.split(data)):
    train_data = data[train_idx]
    val_data = data[val_idx]
    # Train model on this fold
```

---

## 🎮 Training Strategies

### 1. Progressive Training

Start với easier task, gradually increase difficulty:

```python
# Stage 1: Simple factual Q&A (200 steps)
# Stage 2: Add reasoning questions (200 steps)
# Stage 3: Add complex scenarios (200 steps)
```

### 2. Curriculum Learning

Order data from easy to hard:

```python
# Sort by complexity:
data_sorted = sorted(data, key=lambda x: len(x['output']))

# Or by difficulty score:
data_sorted = sorted(data, key=lambda x: x['difficulty'])
```

### 3. Multi-stage Fine-tuning

```python
# Stage 1: General domain adaptation
dataset_general = load_general_data()
train(model, dataset_general, epochs=2)

# Stage 2: Specific task fine-tuning
dataset_specific = load_specific_data()
train(model, dataset_specific, epochs=3)
```

### 4. Learning Rate Schedules

**Warmup + Linear Decay:**

```python
warmup_steps = 100
lr_scheduler_type = "linear"
```

**Warmup + Cosine:**

```python
warmup_steps = 100
lr_scheduler_type = "cosine"
```

**Cyclic Learning Rate:**

```python
lr_scheduler_type = "cosine_with_restarts"
num_cycles = 3
```

### 5. Mixed Precision Training

```python
# Automatic mixed precision
fp16 = True  # or bf16 = True

# Reduces memory and speeds up training ~2x
```

### 6. Gradient Checkpointing

```python
use_gradient_checkpointing = True

# Trade compute for memory:
# - Recompute activations during backward pass
# - Saves memory (can use larger batch sizes)
# - ~20% slower but worth it
```

---

## 📈 Monitoring & Debugging

### Metrics to Track

#### 1. **Training Loss**

```python
# Should decrease over time
# Típical pattern:
#   0-100 steps: Rapid decrease
#   100-300 steps: Steady decrease
#   300+ steps: Plateau

# Red flags:
- Loss not decreasing → LR too low
- Loss spiking → LR too high
- Loss oscillating → Batch size too small
- Loss = NaN → Gradient explosion
```

#### 2. **Validation Loss**

```python
# Should track training loss
# If diverges → Overfitting

# Overfitting signs:
train_loss decreases, val_loss increases
→ Stop training or add regularization
```

#### 3. **Learning Rate**

```python
# Track LR changes over time
# Should follow scheduler pattern
```

#### 4. **Gradient Norm**

```python
# Monitor gradient magnitudes
# Very high (>10) → Gradient explosion
# Very low (<0.001) → Gradient vanishing
```

#### 5. **Perplexity**

```python
perplexity = exp(loss)

# Lower = better
# Typical ranges:
# - Good: 1-3
# - OK: 3-10
# - Bad: >10
```

### Logging Setup

```python
# TensorBoard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/experiment_1')

# Weights & Biases
import wandb
wandb.init(project="llm-finetuning")

# In training args:
report_to = "tensorboard"  # or "wandb"
logging_steps = 10
```

### Visualization

```python
import matplotlib.pyplot as plt

# Plot training loss
plt.plot(train_losses)
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# Plot LR schedule
plt.plot(learning_rates)
plt.xlabel('Steps')
plt.ylabel('Learning Rate')
plt.title('LR Schedule')
plt.show()
```

### Early Stopping

```python
from transformers import EarlyStoppingCallback

# Stop if no improvement for N evaluations
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=0.01
)

trainer = Trainer(
    ...
    callbacks=[early_stopping]
)
```

### Checkpointing

```python
# Save best model based on metric
save_strategy = "steps"
save_steps = 100
save_total_limit = 3  # Keep only 3 best checkpoints

# Or based on evaluation:
evaluation_strategy = "steps"
eval_steps = 100
load_best_model_at_end = True
metric_for_best_model = "eval_loss"
```

---

## ✅ Best Practices

### 1. Start Simple

```python
# Begin with:
- Small model (3-7B)
- Small dataset (100-500 samples)
- Low rank LoRA (r=8)
- Conservative LR (1e-4)

# Then scale up:
- Increase dataset
- Increase rank
- Tune hyperparameters
```

### 2. Version Control Everything

```python
# Track:
- Training scripts
- Hyperparameters (save to config.yaml)
- Dataset versions
- Model checkpoints

# Use Git + DVC/LFS:
git add train.py config.yaml
dvc add dataset.json
git commit -m "Experiment 1: baseline"
```

### 3. Experiment Tracking

```python
# Document each experiment:
experiment = {
    "id": "exp_001",
    "date": "2024-01-15",
    "model": "gemma-2-9b",
    "dataset_size": 300,
    "hyperparams": {
        "lr": 2e-4,
        "r": 16,
        "epochs": 3
    },
    "results": {
        "final_loss": 0.45,
        "val_loss": 0.52,
        "quality": "Good on most questions"
    },
    "notes": "Overfitting after epoch 2"
}
```

### 4. Test Continuously

```python
# Test after every N steps:
test_prompts = [
    "Test question 1",
    "Test question 2",
    "Edge case question"
]

def test_model(model, prompts):
    for prompt in prompts:
        output = generate(model, prompt)
        print(f"Q: {prompt}")
        print(f"A: {output}\n")

# Run every 100 steps
if step % 100 == 0:
    test_model(model, test_prompts)
```

### 5. Hyperparameter Search

**Grid Search:**

```python
learning_rates = [1e-4, 2e-4, 5e-4]
ranks = [8, 16, 32]

for lr in learning_rates:
    for r in ranks:
        train(lr=lr, rank=r)
        evaluate()
```

**Random Search:**

```python
import random

for i in range(10):
    lr = random.uniform(1e-5, 5e-4)
    r = random.choice([8, 16, 32, 64])
    train(lr=lr, rank=r)
```

**Bayesian Optimization:**

```python
from optuna import create_study

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 5e-4)
    r = trial.suggest_int('rank', 8, 64)

    loss = train_and_evaluate(lr=lr, rank=r)
    return loss

study = create_study(direction='minimize')
study.optimize(objective, n_trials=20)
```

### 6. Data Quality over Quantity

**Quality checklist:**

```python
def check_data_quality(sample):
    checks = {
        "has_instruction": len(sample['instruction']) > 0,
        "has_output": len(sample['output']) > 0,
        "output_length": 10 < len(sample['output']) < 1000,
        "no_duplicates": sample not in seen_samples,
        "proper_format": is_valid_format(sample)
    }
    return all(checks.values())
```

### 7. Regularization Stack

```python
# Use multiple regularization techniques:
config = {
    "weight_decay": 0.01,        # L2 regularization
    "lora_dropout": 0.05,        # Dropout
    "max_grad_norm": 1.0,        # Gradient clipping
    "early_stopping": True,      # Stop before overfitting
    "data_augmentation": True    # More diverse data
}
```

### 8. Document Everything

```python
# Create README.md for each experiment:
"""
# Experiment: Gemma-2-9B Personal Assistant

## Objective
Fine-tune Gemma-2-9B to answer questions about Nguyen Tuan Thanh

## Dataset
- Size: 300 samples
- Format: Q&A pairs
- Source: CV + GitHub projects
- Quality: Manual review

## Hyperparameters
- Model: unsloth/gemma-2-9b-bnb-4bit
- LoRA rank: 16
- LoRA alpha: 16
- Learning rate: 2e-4
- Batch size: 2
- Gradient accumulation: 4
- Steps: 200

## Results
- Final train loss: 0.35
- Sample outputs: [link]
- Quality assessment: 9/10

## Issues & Solutions
- Issue: Overfitting after 150 steps
- Solution: Added weight decay 0.01

## Next Steps
- [ ] Increase dataset to 500 samples
- [ ] Try rank 32
- [ ] Add more diverse questions
"""
```

### 9. Gradual Scaling

```python
# Phase 1: Proof of concept
- 50 samples
- Quick training (50 steps)
- Verify pipeline works

# Phase 2: Small scale
- 100-200 samples
- Proper training (100-200 steps)
- Tune hyperparameters

# Phase 3: Full scale
- 300+ samples
- Long training (300-500 steps)
- Final model
```

### 10. Evaluation Beyond Loss

```python
# Quantitative metrics:
- BLEU score (for translation)
- ROUGE score (for summarization)
- Exact match (for Q&A)
- F1 score

# Qualitative evaluation:
- Manual review of outputs
- A/B testing
- User feedback
- Edge case testing
```

---

## 🐛 Common Issues & Solutions

### Issue 1: Out of Memory (OOM)

**Symptoms:**

```
RuntimeError: CUDA out of memory
```

**Solutions:**

```python
# Solution 1: Reduce batch size
per_device_train_batch_size = 1  # Minimum

# Solution 2: Increase gradient accumulation
gradient_accumulation_steps = 8

# Solution 3: Use 4-bit quantization
load_in_4bit = True

# Solution 4: Enable gradient checkpointing
use_gradient_checkpointing = True

# Solution 5: Reduce sequence length
max_seq_length = 1024  # Instead of 2048

# Solution 6: Reduce LoRA rank
r = 8  # Instead of 16

# Nuclear option: Smaller model
model_name = "unsloth/gemma-2-2b-bnb-4bit"
```

### Issue 2: Loss Not Decreasing

**Symptoms:**

```
Loss stays constant or decreases very slowly
```

**Solutions:**

```python
# Solution 1: Increase learning rate
learning_rate = 5e-4  # From 1e-4

# Solution 2: Check data format
# Ensure EOS token is added
text = prompt + tokenizer.eos_token

# Solution 3: More training steps
max_steps = 500  # From 200

# Solution 4: Increase LoRA rank
r = 32  # From 16

# Solution 5: Check if data is too hard
# Start with simpler examples
```

### Issue 3: Loss Exploding (→ NaN)

**Symptoms:**

```
Loss suddenly becomes NaN
or jumps to very high values
```

**Solutions:**

```python
# Solution 1: Reduce learning rate
learning_rate = 1e-4  # From 5e-4

# Solution 2: Enable gradient clipping
max_grad_norm = 1.0

# Solution 3: Use BFloat16 instead of Float16
bf16 = True
fp16 = False

# Solution 4: Reduce batch size
per_device_train_batch_size = 1

# Solution 5: Add warmup
warmup_steps = 50
```

### Issue 4: Overfitting

**Symptoms:**

```
Train loss decreases but val loss increases
Model memorizes instead of generalizes
```

**Solutions:**

```python
# Solution 1: Add weight decay
weight_decay = 0.1  # From 0.01

# Solution 2: Add dropout
lora_dropout = 0.1  # From 0.0

# Solution 3: Early stopping
early_stopping_patience = 3

# Solution 4: Reduce training steps
max_steps = 100  # From 300

# Solution 5: More data
# Augment dataset or collect more samples

# Solution 6: Reduce model capacity
r = 8  # From 16
```

### Issue 5: Model Forgets Base Knowledge

**Symptoms:**

```
Model good at new task but bad at general tasks
Catastrophic forgetting
```

**Solutions:**

```python
# Solution 1: Lower learning rate
learning_rate = 1e-4  # More conservative

# Solution 2: Reduce training steps
max_steps = 100  # Don't overtrain

# Solution 3: Mix general data
# Add 10-20% general Q&A to dataset

# Solution 4: Use LoRA instead of full fine-tuning
# LoRA less likely to forget

# Solution 5: Periodic evaluation
# Test general capabilities regularly
```

### Issue 6: Slow Training

**Symptoms:**

```
Training taking too long
Low GPU utilization
```

**Solutions:**

```python
# Solution 1: Increase batch size
per_device_train_batch_size = 4

# Solution 2: Enable mixed precision
bf16 = True  # or fp16 = True

# Solution 3: Reduce gradient accumulation
gradient_accumulation_steps = 1

# Solution 4: Optimize data loading
dataset_num_proc = 4

# Solution 5: Use faster optimizer
optim = "adamw_8bit"

# Solution 6: Profile code
# Find bottlenecks and optimize
```

### Issue 7: Poor Output Quality

**Symptoms:**

```
Model generates incorrect or low-quality responses
```

**Solutions:**

```python
# Solution 1: More training
max_steps = 500

# Solution 2: Better data quality
# Clean and verify dataset

# Solution 3: Increase model capacity
r = 32  # Higher rank

# Solution 4: Better prompts
# Improve prompt template

# Solution 5: Tune generation params
temperature = 0.7
top_p = 0.9
repetition_penalty = 1.1

# Solution 6: More diverse data
# Add varied examples
```

### Issue 8: Model Outputs Repetitive Text

**Symptoms:**

```
Model repeats same phrases
or gets stuck in loops
```

**Solutions:**

```python
# Solution 1: Adjust generation params
repetition_penalty = 1.2  # Penalize repetition
no_repeat_ngram_size = 3  # No 3-gram repeats

# Solution 2: Temperature tuning
temperature = 0.8  # Higher for diversity

# Solution 3: Check for repeated patterns in data
# Remove duplicate samples

# Solution 4: Add EOS token properly
text = prompt + tokenizer.eos_token

# Solution 5: Reduce max_length
max_new_tokens = 128  # Prevent running too long
```

---

## 📚 Additional Resources

### Research Papers

1. **LoRA: Low-Rank Adaptation of Large Language Models**

   - https://arxiv.org/abs/2106.09685
   - Original LoRA paper

2. **QLoRA: Efficient Finetuning of Quantized LLMs**

   - https://arxiv.org/abs/2305.14314
   - 4-bit quantization + LoRA

3. **Parameter-Efficient Transfer Learning for NLP**
   - https://arxiv.org/abs/1902.00751
   - Adapter methods

### Tools & Libraries

```python
# Unsloth - Fast fine-tuning
# https://github.com/unslothai/unsloth

# Hugging Face Transformers
# https://github.com/huggingface/transformers

# PEFT - Parameter-Efficient Fine-Tuning
# https://github.com/huggingface/peft

# BitsAndBytes - Quantization
# https://github.com/TimDettmers/bitsandbytes

# TRL - Transformer Reinforcement Learning
# https://github.com/huggingface/trl
```

### Courses & Tutorials

1. **Hugging Face Course**

   - https://huggingface.co/course
   - Free, comprehensive

2. **Fast.ai Deep Learning Course**

   - https://course.fast.ai/
   - Practical approach

3. **Stanford CS224N: NLP with Deep Learning**
   - https://web.stanford.edu/class/cs224n/
   - Theoretical foundation

### Communities

- Hugging Face Discord
- r/LocalLLaMA (Reddit)
- Unsloth Discord
- AI Alignment Forum

---

## 🎓 Glossary

### Key Terms

**Adapter:** Small trainable modules added to frozen model

**Batch Size:** Number of samples processed together

**Catastrophic Forgetting:** Model forgets previous knowledge

**Checkpoint:** Saved model state at a point in time

**Convergence:** When loss stops improving (plateau)

**Dropout:** Randomly disable neurons to prevent overfitting

**Early Stopping:** Stop training when validation stops improving

**Embedding:** Vector representation of text

**Epoch:** One complete pass through dataset

**Fine-tuning:** Adapt pre-trained model to specific task

**Gradient:** Direction and magnitude of weight updates

**Gradient Accumulation:** Accumulate gradients over steps

**Gradient Clipping:** Limit gradient magnitude

**Hyperparameter:** Configuration value (LR, batch size, etc.)

**Inference:** Using model to generate predictions

**Learning Rate:** Step size for weight updates

**LoRA:** Low-Rank Adaptation for efficient fine-tuning

**Loss:** Measure of prediction error

**Overfitting:** Model memorizes training data, poor generalization

**Perplexity:** Measure of model uncertainty (lower = better)

**Quantization:** Reduce precision of weights (32-bit → 4-bit)

**Regularization:** Techniques to prevent overfitting

**Scheduler:** Change learning rate during training

**Token:** Basic unit of text (word/subword)

**Tokenizer:** Convert text ↔ numbers

**Underfitting:** Model too simple, poor performance

**Validation Set:** Data held out to check generalization

**Warmup:** Gradually increase learning rate at start

**Weight Decay:** L2 regularization penalty

---

## 📋 Quick Reference Cheat Sheet

### Default Configuration (Good Starting Point)

```python
# Model
model_name = "unsloth/gemma-2-9b-bnb-4bit"
max_seq_length = 2048
load_in_4bit = True

# LoRA
r = 16
lora_alpha = 16
lora_dropout = 0.0
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training
learning_rate = 2e-4
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
max_steps = 200
warmup_steps = 10

# Optimization
optim = "adamw_8bit"
weight_decay = 0.01
lr_scheduler_type = "linear"
max_grad_norm = 1.0

# Precision
bf16 = True  # If supported
fp16 = False

# Logging
logging_steps = 5
save_steps = 50
```

### Troubleshooting Quick Guide

| Problem             | Quick Fix              |
| ------------------- | ---------------------- |
| OOM                 | Reduce batch size to 1 |
| Loss not decreasing | Increase LR to 5e-4    |
| Loss → NaN          | Reduce LR to 1e-4      |
| Overfitting         | Add weight_decay = 0.1 |
| Slow training       | Enable bf16/fp16       |
| Poor quality        | Train more steps       |

### Commands Cheat Sheet

```bash
# Install
pip install unsloth transformers trl

# Check GPU
nvidia-smi

# Monitor training
tensorboard --logdir runs/

# Kill process if stuck
pkill -9 python
```

---

## 🎯 Conclusion

Fine-tuning LLMs is both art and science. Key principles:

1. **Start simple, iterate**
2. **Data quality > quantity**
3. **Monitor closely**
4. **Document everything**
5. **Be patient**

Happy fine-tuning! 🚀

---

_Last updated: 2024_
_Created by: Nguyen Tuan Thanh_
_For questions: tuanthanh2kk4@gmail.com_
