# 🎛️ Deep Learning — How to Find the Right Parameters
### A Decision Guide for Every Design Choice

> **Purpose:** Every time you build a deep learning model, you face dozens of decisions. This guide gives you a systematic framework so you stop guessing and start reasoning.

---

## The Master Decision Flow

```
1. Define task type  → determines architecture, output layer, loss function
2. Design network    → choose depth, width, activations
3. Choose optimizer  → Adam/SGD/AdamW + learning rate + schedule
4. Add regularization → dropout, batchnorm, weight decay, early stopping
5. Set training params → batch size, epochs
6. Encode inputs     → scaling, encoding, tokenization
7. Train → diagnose → tune → repeat
```

---

---

# SECTION 1 — Task Type: The First Decision

---

## 1.1 Why You Must Define the Task First

Every other decision flows from your task type. Get this wrong and nothing else can save you.

| Task | Example | Architecture | Loss Function |
|------|---------|--------------|---------------|
| Binary classification | Spam detection, fraud | MLP / CNN / LSTM | BCEWithLogitsLoss |
| Multi-class (1 of N) | Digit recognition, emotion | MLP / CNN | CrossEntropyLoss |
| Multi-label (M of N) | Image tags | MLP / CNN | BCEWithLogitsLoss |
| Regression | House price, temperature | MLP | MSELoss / HuberLoss |
| Image classification | Cat vs Dog | CNN | CrossEntropyLoss |
| Text classification | Sentiment analysis | LSTM / Transformer | CrossEntropyLoss |
| Sequence to sequence | Translation | Transformer | CrossEntropyLoss |
| Time series | Stock price | LSTM / 1D-CNN | MSELoss |

---

## 1.2 Output Layer + Loss Function — The Exact Rules

The output layer and loss function are linked. You must get this pair right.

```python
# Binary Classification (2 classes)
# Output: single raw logit
# Loss: combines sigmoid + binary cross-entropy
nn.Linear(hidden, 1)
nn.BCEWithLogitsLoss()

# Multi-class Classification (N classes)
# Output: N raw logits (one per class)
# Loss: internally applies log-softmax
nn.Linear(hidden, num_classes)
nn.CrossEntropyLoss()

# Regression (continuous output, unbounded)
# Output: single raw value, no activation
nn.Linear(hidden, 1)
nn.MSELoss()  # or nn.HuberLoss()

# Regression (output must be positive)
# Output: raw value passed through ReLU
nn.Sequential(nn.Linear(hidden, 1), nn.ReLU())
nn.MSELoss()
```

**Three rules that will save you from painful debugging:**

1. **Never** apply Softmax before CrossEntropyLoss — it applies log-softmax internally
2. **Never** apply Sigmoid before BCEWithLogitsLoss — same problem
3. For regression: decide whether your output is bounded or unbounded, and choose the activation accordingly

---

---

# SECTION 2 — Network Depth: How Many Layers?

---

## 2.1 What Depth Does

Each additional hidden layer allows the network to learn a more abstract representation of the data. Early layers detect simple patterns; deeper layers combine those into complex patterns.

**But there are trade-offs:**

| More Depth | Less Depth |
|-----------|-----------|
| Can learn more complex patterns | Faster training, easier to debug |
| Needs more data | Works well on simpler tasks |
| Harder to train (vanishing gradient) | Lower risk of overfitting |
| Use BatchNorm + skip connections | — |

---

## 2.2 Depth Guidelines by Task

| Dataset Complexity | Hidden Layers | Example |
|-------------------|--------------|---------|
| Very simple / nearly linear | 1–2 | Iris classification |
| Moderate | 2–4 | Tabular fraud detection |
| High complexity | 4–8 | Medical imaging tabular |
| Very high (images) | 8+ (use ResNet skip connections) | ImageNet |

**Starting strategy:**
```
Start with 2 hidden layers.
If model underfits → add depth.
Never start deep and try to prune. Start shallow and build up.
```

---

---

# SECTION 3 — Network Width: How Many Neurons?

---

## 3.1 What Width Does

Width (number of neurons per layer) determines how much the network can represent at each level of abstraction. Too few → underfitting. Too many → overfitting and slow training.

---

## 3.2 Width Patterns

**The Funnel Pattern (most common for classification):**
```
Input (100 features) → 256 → 128 → 64 → Output (10 classes)
```
Each layer compresses information, forcing the network to learn compact representations.

**The Constant Pattern (for autoencoders, transformers):**
```
Input → 256 → 256 → 256 → Output
```
No forced compression — useful when you want to preserve sequence information.

**Always use powers of 2:**
```
32, 64, 128, 256, 512, 1024
```
GPU memory is optimized for these sizes. Arbitrary sizes like 100 or 300 waste memory.

---

## 3.3 Starting Widths by Dataset Size

```
Small dataset (< 10k samples):   [64, 32]
Medium dataset (10k–100k):       [256, 128, 64]
Large dataset (100k+):           [512, 256, 128, 64]
Image classifier head:           [512, 256]
```

---

## 3.4 Width vs Depth — Which to Increase When

```
Model is UNDERFITTING (high train loss):
  → First: increase width (more neurons)
  → Then: increase depth (more layers)

Model is OVERFITTING (val loss >> train loss):
  → Do NOT increase width or depth
  → Add Dropout, weight decay, or more data first
  → If still overfitting: reduce model size
```

---

---

# SECTION 4 — Activation Functions

---

## 4.1 Why Activation Functions Exist

Without non-linear activation functions, stacking any number of linear layers produces only a single linear transformation. Non-linearity is what allows neural networks to learn complex patterns.

---

## 4.2 Every Activation Function Explained

### ReLU — Rectified Linear Unit
```
f(x) = max(0, x)
```
**Use for:** All hidden layers in MLP and CNN (default choice).

**Advantages:** Fast to compute, no vanishing gradient for positive inputs, works in almost all cases.

**Disadvantage:** "Dead neurons" — if a neuron's pre-activation is always negative, it always outputs 0 and never learns. Usually not a problem with good initialization.

---

### Leaky ReLU
```
f(x) = x  if x > 0,  else  0.01 × x
```
**Use for:** When you observe many dead ReLU neurons during training.

**Why it works:** The small negative slope (0.01) keeps neurons alive even when pre-activation is negative.

---

### GELU — Gaussian Error Linear Unit
```
f(x) ≈ x × Φ(x)     where Φ is the Gaussian CDF
```
**Use for:** Transformers, BERT, GPT, and any modern NLP architecture.

**Why:** Smoother gradient than ReLU, performs better in attention mechanisms.

---

### Sigmoid
```
f(x) = 1 / (1 + e^{-x})     → output in (0, 1)
```
**Use ONLY for:** Binary classification output layer.

**Never use in hidden layers:** Saturates (output near 0 or 1) → near-zero gradient → vanishing gradient problem.

---

### Softmax
```
f(x_i) = e^{x_i} / Σ e^{x_j}     → outputs sum to 1
```
**Use ONLY for:** Multi-class output layer when you need probabilities.

**Never use in hidden layers.** And remember: CrossEntropyLoss handles Softmax internally — do not apply it manually when using that loss function.

---

## 4.3 Quick Decision Table

| Location | Use |
|----------|-----|
| Hidden layer in MLP or CNN | ReLU |
| Hidden layer in Transformer | GELU |
| Hidden layer in LSTM | Tanh (built-in) |
| Output: binary classification | None (use BCEWithLogitsLoss) |
| Output: multi-class | None (use CrossEntropyLoss) |
| Output: regression, unbounded | None (linear) |
| Output: regression, 0 to 1 range | Sigmoid |
| Output: regression, positive only | ReLU |

---

---

# SECTION 5 — Optimizers

---

## 5.1 What an Optimizer Does

An optimizer uses the computed gradients to update model weights. The choice of optimizer affects how fast the model learns and how well it generalizes.

---

## 5.2 The Four Main Optimizers

### Adam (Adaptive Moment Estimation)
```python
optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
```
**Use for:** Default for most tasks — NLP, tabular, image classification experiments.

**How it works:** Maintains a running estimate of both the first moment (mean of gradients) and second moment (variance of gradients), adapting the learning rate per parameter.

**Advantage:** Works well out of the box with minimal tuning.

---

### AdamW (Adam with Weight Decay Fix)
```python
optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```
**Use for:** Transformer fine-tuning, BERT, GPT — any modern NLP.

**Why different from Adam:** In standard Adam, weight decay is applied to the adjusted gradient, which is incorrect. AdamW decouples weight decay from the gradient update, which works as intended.

---

### SGD with Momentum
```python
optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
```
**Use for:** Image classification CNNs, especially when best final accuracy is needed.

**How momentum works:** Accumulates a velocity vector in directions of persistent gradients, accelerating convergence and dampening oscillations.

**Trade-off:** Converges slower than Adam but often generalizes better on image tasks with a good learning rate schedule.

---

### RMSprop
```python
optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99)
```
**Use for:** RNNs, LSTMs — handles noisy/non-stationary gradients well.

---

## 5.3 Quick Selection

```
New experiment, any task         → Adam (lr=1e-3)
CNN image classification         → SGD (lr=0.01, momentum=0.9)
Transformer / BERT fine-tuning   → AdamW (lr=2e-5)
LSTM / RNN                       → RMSprop (lr=1e-3)
```

---

---

# SECTION 6 — Learning Rate

---

## 6.1 The Most Important Hyperparameter

Learning rate is the single most impactful hyperparameter in deep learning. Setting it wrong causes failure:

```
Too high → loss explodes, oscillates, or diverges from the start
Too low  → training is painfully slow, gets stuck in local minima
Just right → smooth convergence to a good solution
```

---

## 6.2 Starting Learning Rate by Optimizer

| Optimizer | Typical Range | Safe Default |
|-----------|--------------|-------------|
| Adam | 1e-4 to 1e-3 | 1e-3 |
| AdamW (fine-tuning LLMs) | 1e-5 to 1e-4 | 2e-5 |
| AdamW (training from scratch) | 1e-4 to 1e-3 | 1e-3 |
| SGD | 0.01 to 0.1 | 0.01 |
| RMSprop | 1e-4 to 1e-3 | 1e-3 |

---

## 6.3 Learning Rate Schedules

A fixed learning rate is rarely optimal. Schedules adapt the LR during training.

### ReduceLROnPlateau (recommended starting point)
```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
# Call each epoch: scheduler.step(val_loss)
# Halves LR when val_loss hasn't improved for 5 epochs
# Use when you don't know training length in advance
```

### CosineAnnealingLR (smooth, popular)
```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
# Smoothly decreases LR following cosine curve
# Good for image classification CNNs
```

### StepLR (simple, predictable)
```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# Multiply LR by gamma every step_size epochs
# Good when you know training length
```

### Warmup + Cosine Decay (Transformer standard)
```python
from transformers import get_cosine_schedule_with_warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=100, num_training_steps=1000
)
# LR increases linearly during warmup, then decays via cosine
```

---

---

# SECTION 7 — Regularization

---

## 7.1 Dropout

**Definition:** **Dropout** randomly sets a fraction of neuron outputs to zero during training, forcing the network to learn redundant representations and preventing over-reliance on any single neuron.

```python
nn.Dropout(p=0.5)   # 50% of neurons zeroed each forward pass during training
```

**Dropout rates by location:**

| Location | Rate |
|----------|------|
| After large FC layers | 0.4–0.5 |
| After small FC layers | 0.2–0.3 |
| After conv layers | 0.1–0.2 (rarely needed) |
| Transformer attention | 0.1 |

**Critical:** Dropout is only active during training.
```python
model.train()   # Dropout ON
model.eval()    # Dropout OFF — always call this before evaluation
```

---

## 7.2 Batch Normalization

**Definition:** **Batch Normalization** normalizes the output of each layer to have zero mean and unit variance across the mini-batch, then applies learnable scale (γ) and shift (β) parameters.

**Formula:**
```
μ_B = mean of batch
σ_B = std of batch
x̂  = (x − μ_B) / (σ_B + ε)     ← normalize
y   = γ × x̂ + β                  ← scale and shift (learned)
```

**Benefits:**
- Allows higher learning rates (more stable gradients)
- Reduces sensitivity to weight initialization
- Acts as regularizer (reduces need for Dropout)
- Makes very deep networks trainable

**Placement:**
```python
# Standard: Conv/Linear → BatchNorm → Activation
nn.Linear(256, 128)
nn.BatchNorm1d(128)   # ← after linear
nn.ReLU()
```

**When to use what:**
| Scenario | Use |
|----------|-----|
| MLP or CNN | BatchNorm |
| Transformer or RNN | LayerNorm |
| Batch size < 8 | LayerNorm (BatchNorm unreliable) |

---

## 7.3 L2 Regularization (Weight Decay)

Adds a penalty proportional to the sum of squared weights to the loss:
```
Total Loss = Task Loss + λ × Σ(w²)
```

In PyTorch, pass it to the optimizer:
```python
optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
# Common values: 1e-5 to 1e-3
# Safe default: 1e-4
```

---

## 7.4 Early Stopping

Stop training when validation loss stops improving. Prevents overfitting and saves compute.

```python
best_val_loss = float('inf')
patience = 10
counter = 0

for epoch in range(MAX_EPOCHS):
    val_loss = evaluate(...)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pt')   # save best
    else:
        counter += 1
    if counter >= patience:
        print("Early stopping"); break

# Always load best model after training
model.load_state_dict(torch.load('best_model.pt'))
```

---

---

# SECTION 8 — Batch Size and Epochs

---

## 8.1 Batch Size

**Definition:** The number of training samples processed before one weight update.

| Batch Size | Gradient Quality | Speed | Memory | Generalization |
|-----------|-----------------|-------|--------|----------------|
| Small (8–32) | Noisy | Slow | Low | Often better |
| Large (128–512) | Accurate | Fast | High | Can be worse |

Small batches introduce noise that acts like regularization and can help escape sharp minima. Large batches are faster but may find sharp minima that generalize poorly.

**Practical guidelines:**
```
Safe default:           32 or 64
Large GPU available:    128 or 256
Transformer fine-tune:  16–32 (high memory cost per sample)
Very small dataset:     8–16
Training unstable:      reduce batch size
Training too slow:      increase batch size
```

---

## 8.2 Epochs

**Always use Early Stopping instead of a fixed epoch count.**

If you must set a fixed value:
| Dataset Size | Typical Epochs |
|-------------|----------------|
| Small (< 5k) | 50–200 |
| Medium (5k–100k) | 20–100 |
| Large (100k+) | 5–30 |
| Transfer learning | 3–10 |

---

---

# SECTION 9 — Data Encoding

---

## 9.1 Categorical Features

**One-Hot Encoding**
```python
pd.get_dummies(df['color'])
# [Red, Blue, Green] → [1,0,0], [0,1,0], [0,0,1]
# Use when: low cardinality (< 15 unique values)
# Avoid when: high cardinality → too many features
```

**Label Encoding**
```python
from sklearn.preprocessing import LabelEncoder
# [Low, Medium, High] → [0, 1, 2]
# Use ONLY when: there is a true ordering
# Never use for unordered categories — implies false ranking
```

**Target Encoding**
```python
df['city_encoded'] = df.groupby('city')['price'].transform('mean')
# Replace category with target mean for that category
# Use when: high cardinality + tree-based models
# Risk: data leakage — always compute on train set only
```

**Learnable Embeddings (Neural Networks)**
```python
nn.Embedding(num_categories, embedding_dim)
# Maps integer ID → dense vector, learned during training
# Use when: high cardinality categoricals in neural networks
# Rule of thumb: embedding_dim = min(50, num_categories // 2 + 1)
```

**Quick reference:**
```
Low cardinality + any model          → One-Hot
True ordinal order                    → Label Encoding
High cardinality + tree model         → Target Encoding
High cardinality + neural network     → Embedding
```

---

## 9.2 Numerical Features — Scaling

**Why scale?** Neural networks are sensitive to feature magnitude. Features on different scales cause gradients to flow unequally — some features dominate the learning process.

**Standard Scaling (Z-score)**
```python
from sklearn.preprocessing import StandardScaler
# x_scaled = (x − mean) / std → output centered at 0, std = 1
# Use when: approximately normal distribution, no heavy outliers
# Default for: most neural network inputs
```

**Min-Max Scaling**
```python
from sklearn.preprocessing import MinMaxScaler
# x_scaled = (x − min) / (max − min) → output in [0, 1]
# Use when: need bounded output
# Problem: sensitive to outliers — one extreme value compresses everything
```

**Robust Scaling**
```python
from sklearn.preprocessing import RobustScaler
# Uses median and IQR instead of mean and std
# Use when: data contains significant outliers
```

**Log Transform**
```python
import numpy as np
df['income_log'] = np.log1p(df['income'])  # log(1 + x), handles zeros
# Use when: right-skewed distribution (income, sales, counts)
# Then apply StandardScaler after
```

**Quick reference:**
```
Normal distribution, no outliers     → StandardScaler
Bounded range needed                  → MinMaxScaler
Outliers present                      → RobustScaler
Right-skewed (income, counts)         → Log → StandardScaler
Binary (0/1 features)                 → No scaling needed
Image pixels (0–255)                  → Divide by 255
```

---

---

# SECTION 10 — Diagnosing and Tuning

---

## 10.1 Reading Training Curves

| Pattern | Diagnosis | Action |
|---------|-----------|--------|
| Both losses decreasing steadily | ✅ Healthy | Keep training |
| Train loss low, val loss rising | 🔴 Overfitting | More Dropout, augment, reduce LR |
| Both losses high and flat | 🔴 Underfitting | Bigger model, higher LR, more epochs |
| Val loss spikes randomly | ⚠️ LR too high | Reduce LR by 10× |
| Val loss better than train loss | Interesting | Often means Dropout is too strong |

---

## 10.2 Tuning Priority Order

Tune one thing at a time, in this order:

```
Priority 1: Learning Rate       — single biggest impact
Priority 2: Architecture        — depth and width
Priority 3: Batch Size          — gradient quality
Priority 4: Dropout Rate        — regularization strength
Priority 5: Weight Decay        — secondary regularization
Priority 6: LR Schedule         — squeeze out last performance
Priority 7: Optimizer           — usually Adam is fine
Priority 8: Activation Function — usually ReLU is fine
```

---

## 10.3 Safe Starting Defaults for Any Problem

```python
# Architecture
hidden_layers = [256, 128, 64]    # funnel pattern
activation    = nn.ReLU()

# Regularization
dropout_rate  = 0.3               # moderate
use_batchnorm = True
weight_decay  = 1e-4

# Training
optimizer     = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler     = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
batch_size    = 64
max_epochs    = 100
early_stopping_patience = 15
```

---

---

# SECTION 11 — Q&A and Practice

---

## 11.1 Conceptual Questions

**Q1: You use CrossEntropyLoss but add Softmax at the end of your model. What goes wrong and how do you fix it?**

> CrossEntropyLoss internally applies log-softmax. Your model applies softmax first, then CrossEntropyLoss applies log-softmax again to values that are already probabilities (in 0–1 range). The log of a small probability like 0.0001 is −9.2 — this creates extreme, incorrect loss values. Fix: remove Softmax from the model's forward method.

**Q2: You double the batch size from 64 to 128 for speed. Should you change the learning rate? How?**

> Yes — the Linear Scaling Rule suggests doubling the learning rate when you double the batch size. With a larger batch, each gradient estimate is more accurate (less noisy), so you can take a larger step. This is an approximation — it works up to a point but very large learning rates still destabilize training.

**Q3: Your model reaches 95% train accuracy but only 52% val accuracy after 50 epochs. Name 3 specific interventions.**

> 1. Add data augmentation (if working with images) — increases effective dataset diversity. 2. Increase Dropout rate from your current value to 0.5 — forces more redundant representations. 3. Add L2 weight decay (`weight_decay=1e-3` in optimizer) — penalizes large weights.

**Q4: What is the difference between BatchNorm and Dropout as regularizers?**

> Dropout works by randomly disabling neurons — forcing the network to not over-rely on any one path through the network. BatchNorm regularizes by adding noise through batch statistics — the mean and variance vary slightly from batch to batch, which acts like a form of data augmentation. They can be used together but BatchNorm often reduces the amount of Dropout needed.

**Q5: You are training a Transformer for text classification. Which optimizer, learning rate, and normalization should you use?**

> Optimizer: AdamW (weight decay is correctly decoupled). Learning rate: 2e-5 for fine-tuning a pretrained model; 1e-3 for training from scratch. Normalization: LayerNorm (not BatchNorm) — Transformers process variable-length sequences, and LayerNorm normalizes over the feature dimension rather than the batch, making it sequence-length independent.

**Q6: What does early stopping actually save you from, and why is it not enough on its own?**

> Early stopping prevents training beyond the point where validation performance peaks — after which the model starts memorizing training examples rather than learning generalizable patterns. It is not enough on its own because it only detects overfitting after it starts. True prevention comes from the right architecture size, Dropout, and weight decay from the beginning.

---

## 11.2 Numerical Exercises

**Exercise 1:** You are training a 3-layer MLP with hidden sizes [256, 128, 64] on a 50-feature input for 10-class classification. Calculate the total number of trainable parameters.

> Input(50) → Linear(50,256) + bias: 50×256 + 256 = 13,056
> Linear(256,128) + bias: 256×128 + 128 = 32,896
> Linear(128,64) + bias: 128×64 + 64 = 8,256
> Linear(64,10) + bias: 64×10 + 10 = 650
> **Total: 54,858 parameters**

**Exercise 2:** You have 10,000 training samples, batch size 64, and train for 20 epochs. How many weight updates (optimizer steps) occur?

> Steps per epoch = ceil(10,000 / 64) = 157 (last batch is partial)
> Total steps = 157 × 20 = **3,140 weight updates**

---

## 11.3 The Project Template

Use this checklist for every deep learning project:

```
☐ 1. Define task (binary/multi-class/regression?)
☐ 2. Choose output layer + loss function (linked pair)
☐ 3. Choose architecture type (MLP/CNN/LSTM?)
☐ 4. Design network (funnel pattern, powers of 2)
☐ 5. Choose activation (ReLU hidden, task-specific output)
☐ 6. Add BatchNorm (after each layer before activation)
☐ 7. Add Dropout (0.3–0.5 after FC layers)
☐ 8. Choose optimizer (Adam default)
☐ 9. Set learning rate (1e-3 default)
☐ 10. Add LR schedule (ReduceLROnPlateau)
☐ 11. Scale numerical features (StandardScaler)
☐ 12. Encode categoricals (one-hot or embedding)
☐ 13. Train with early stopping
☐ 14. Plot loss curves, diagnose, and tune
```

---
