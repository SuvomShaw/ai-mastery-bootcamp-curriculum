# Transfer Learning: ResNet, VGG & Fine-Tuning
###  Notes — Deep Learning / Computer Vision

---

---

## Table of Contents

1. [Why Transfer Learning Exists](#1-why-transfer-learning-exists)
2. [How a CNN Learns Features — The Hierarchy](#2-how-a-cnn-learns-features--the-hierarchy)
3. [The Two Strategies: Feature Extraction vs Fine-Tuning](#3-the-two-strategies-feature-extraction-vs-fine-tuning)
4. [VGG — The Architecture That Is Easy to Understand](#4-vgg--the-architecture-that-is-easy-to-understand)
5. [ResNet — Why Skip Connections Changed Everything](#5-resnet--why-skip-connections-changed-everything)
6. [VGG vs ResNet — Which One and When](#6-vgg-vs-resnet--which-one-and-when)
7. [The Full Workflow — What You Actually Do in a Project](#7-the-full-workflow--what-you-actually-do-in-a-project)
8. [PyTorch Code — The Patterns You Will Use](#8-pytorch-code--the-patterns-you-will-use)
9. [Domain Gap and Catastrophic Forgetting](#9-domain-gap-and-catastrophic-forgetting)
10. [Mistakes to Avoid](#10-mistakes-to-avoid)
11. [Evaluation — Beyond Accuracy](#11-evaluation--beyond-accuracy)

---

## 1. Why Transfer Learning Exists

### The Problem

Imagine you are given a classification task — detect plant disease from leaf photos, classify defective products on a factory line, or label chest X-rays. And you have maybe 2,000 labeled images.

Should you build a deep CNN from scratch?

**No.** Here is why.

A deep CNN has millions of parameters. If all of them are initialized randomly and you only have 2,000 images to train on, the network will memorize your training data rather than learning anything general. You will see high training accuracy and poor validation accuracy. That is overfitting — and it is almost guaranteed at that scale.

### The Solution

Transfer learning says: **someone has already done the hard work of learning visual features from millions of images. Take those learned features and adapt them to your problem. Do not start from zero.**

A model pretrained on ImageNet — 1.2 million images, 1000 classes — has already learned how to see. It knows edges, textures, shapes, object parts. You do not need to teach it those things again. You just need to redirect that knowledge toward your task.

### Why Not Just Train From Scratch?

| Reason | What Happens in Practice |
|---|---|
| Too few images | Model memorizes training set, fails on validation |
| Too many parameters | Optimizer cannot converge reliably |
| Slow training | Takes much longer with no guarantee of improvement |
| Wasteful | You are relearning edge detection, which is already solved |
| High cost | Compute and data labeling are expensive |

### How Industry Teams Actually Work

In practice, the mindset is:

1. Start from a strong pretrained backbone
2. Adapt only what needs to change
3. Validate at every step before adding complexity
4. Fine-tune only as much as your data justifies

> **The core idea:** Transfer Learning = Take knowledge learned on a source task → Adapt it to your target task. The labels differ. The visual knowledge transfers anyway.

---

## 2. How a CNN Learns Features — The Hierarchy

This is the insight that makes transfer learning work, so read this carefully.

When a CNN is trained on a large image dataset, the layers do not all learn the same kind of thing. They learn at different levels of abstraction:

```
Input Image
    │
    ▼
Early Layers ──── Edges, corners, gradients, colour transitions
    │              (very general — useful for ANY visual task)
    ▼
Middle Layers ─── Textures, curves, repeating patterns, colour blobs
    │              (moderately general)
    ▼
Deep Layers ───── Object parts, semantic shapes, fur, wheels, eyes
    │              (more task-specific)
    ▼
Final Head ─────── Source-task classes (ImageNet labels)
                   ← this is what you REPLACE
```

### What This Means for Transfer Learning

**The earlier the layer, the more general its features.**
**The later the layer, the more task-specific.**

Early layers learn edges. Edges appear in dog photos, X-rays, satellite images, and factory floor cameras alike. You do not need to relearn edge detection for every new dataset. That knowledge transfers.

The final classification head, however, was built specifically for ImageNet's 1000 classes. Your problem has different classes. That layer must be replaced.

This is why:
- We **freeze** early layers — they are already useful as-is
- We **replace** the final head — it is task-specific to the wrong task
- We **optionally fine-tune** deeper layers — they may need some adaptation for your domain

---

## 3. The Two Strategies: Feature Extraction vs Fine-Tuning

### Strategy A — Feature Extraction

You freeze the entire pretrained backbone. Nothing in it updates during training. You attach a new classification head on top, and only that head trains.

The pretrained model is just being used as a feature extractor — you are borrowing its eyes, not retraining them.

```
Pretrained Backbone
┌──────────────────────────────────────┐
│  Early layers  │  Mid layers  │  Deep │  ← ALL FROZEN (no gradient updates)
└──────────────────────────────────────┘
                        │
                        ▼
              New Classification Head    ← ONLY THIS TRAINS
              (randomly initialized,
               sized for your classes)
```

**When to use it:**
- Your dataset is small (a few hundred to a few thousand images)
- The domain is reasonably similar to natural photographs
- You want a fast, stable baseline to understand where you stand
- You want to minimize overfitting risk

This should be your **first step on every project**. Always.

---

### Strategy B — Fine-Tuning

After feature extraction gives you a baseline, you can selectively unfreeze some of the deeper layers and let them slowly adapt to your data.

```
Pretrained Backbone
┌────────────────────────────────────────────┐
│  Early layers 🔒  │  Mid 🔒  │  Deep ⚡  │
└────────────────────────────────────────────┘
                          │
                          ▼
              New Classification Head ⚡
```

The key word is **slowly**. You use a very small learning rate for the unfrozen pretrained layers. Those weights already contain useful knowledge — you want to adjust them, not overwrite them.

**When to use it:**
- Your feature extraction baseline has plateaued
- You have enough data to justify deeper adaptation
- The domain gap between ImageNet and your task is significant

**Important:** Fine-tuning means *correction on top of a good starting point*, not retraining from zero. The weights are not random. Treat them carefully.

---

### Comparison

| | Feature Extraction | Fine-Tuning |
|---|---|---|
| Backbone | Fully frozen | Later layers unfrozen |
| What trains | New head only | Head + selected deeper layers |
| Learning rate | Standard (1e-3) | Small for pretrained (1e-5), larger for head (1e-4) |
| Best for | Small dataset, similar domain | Medium–large dataset, bigger domain gap |
| Overfitting risk | Low | Medium — requires careful validation |
| First step? | ✅ Always start here | Only after baseline is established |

### How Much to Unfreeze?

Use this as a decision guide, not a rigid rule:

```
Dataset Size × Domain Similarity
         │
    ┌────┴──────────────────────────────┐
    │                                   │
Small + Similar         Medium / Moderate gap         Large / Big gap
    │                          │                            │
Freeze backbone          Unfreeze last             Broader fine-tuning
Train head only          1–2 blocks                Careful validation
(~500–2K images)        (~2K–20K images)           (large dataset)
```

---

## 4. VGG — The Architecture That Is Easy to Understand

### What It Is

VGG was introduced by Oxford's Visual Geometry Group in 2014. The idea is deliberately simple: use small 3×3 convolution filters, stack them in blocks, halve the spatial size with max pooling after each block, double the number of channels as you go deeper, then flatten and classify with fully connected layers at the end.

That is the entire architecture. There is no branching, no residual connections, no clever tricks. Just depth and repetition.

### VGG16 Structure

```
Input: 224×224×3
│
├── Block 1:  Conv-64 → Conv-64 → MaxPool          → 112×112×64
│
├── Block 2:  Conv-128 → Conv-128 → MaxPool         → 56×56×128
│
├── Block 3:  Conv-256 → Conv-256 → Conv-256 → MaxPool  → 28×28×256
│
├── Block 4:  Conv-512 → Conv-512 → Conv-512 → MaxPool  → 14×14×512
│
├── Block 5:  Conv-512 → Conv-512 → Conv-512 → MaxPool  → 7×7×512
│
├── Flatten
├── FC-4096 → ReLU → Dropout
├── FC-4096 → ReLU → Dropout
└── FC-1000 → Softmax          ← replace this layer for your task
```

Every convolution uses a 3×3 kernel with padding to preserve spatial size within a block. Max pooling with stride 2 halves the spatial dimensions. The number of channels doubles from 64 → 128 → 256 → 512.

**VGG-16 has ~138 million parameters.** Most of those live in the three fully connected layers at the end.

### Why VGG Is Good for Learning

- You can trace every layer without getting confused — it is a straight stack
- Every block does the same thing, just with more channels
- Easy to see how depth builds richer representations
- Easy to replace the head — just swap `classifier[6]`

### Why VGG Is Not Your First Choice in Practice

- 138M parameters is very heavy compared to modern alternatives
- The large FC block at the end is inefficient — high memory, slow inference
- ResNet achieves better accuracy with fewer parameters
- No one would start a new production project with VGG today

VGG is genuinely useful for **understanding CNNs** and as a **classical baseline**. Once you understand VGG, everything else becomes easier to appreciate.

---

## 5. ResNet — Why Skip Connections Changed Everything

### The Problem Before ResNet

Deeper networks should learn richer representations — more layers, more capacity, better results. That was the theory.

In practice, making plain CNNs deeper than about 20–30 layers caused training to degrade. Not just overfit — actually get worse. Even on training data. This is called the **degradation problem** and it stumped researchers for years.

The cause is the **vanishing gradient problem**. During backpropagation, gradients are multiplied through every layer. After 50+ layers of multiplication, the gradient reaching early layers becomes so tiny that those layers stop updating meaningfully.

### The ResNet Solution — Residual Connections

ResNet (He et al., 2015) solved this with one elegant change: instead of asking each block to learn a complete new transformation, let the block learn only the *correction* on top of the input.

Every block now computes:

```
Output = F(x) + x
```

Where:
- `x` is the input passing through a **skip connection** (identity shortcut)
- `F(x)` is the transformation the block learns (Conv → BN → ReLU → Conv → BN)
- The `+` adds them element-wise

```
Input x ──────────────────────────────────┐
    │                                      │  (skip / identity path)
    ▼                                      │
Conv → BN → ReLU                          │
    │                                      │
Conv → BN                                 │
    │                                      │
    └──────────── Add ←────────────────────┘
                   │
                 ReLU
                   │
               Output: F(x) + x
```

### Why This Works

Think of it this way: if a block has nothing useful to contribute, it just sets `F(x) ≈ 0` and the input passes through unchanged via the skip connection. The block only activates when it actually has something to add.

This has two major effects:

1. **Gradients flow directly backward** through the skip connection, bypassing the weight layers. Early layers receive meaningful gradients even in 100-layer networks.
2. **Optimization becomes much easier** — the block learns a correction rather than a full mapping from scratch.

The result: ResNet-50, ResNet-101, ResNet-152 all train reliably. Before ResNet, training a 152-layer network was essentially impossible.

### ResNet Variants

| Model | Parameters | Design | When to Use |
|---|---|---|---|
| ResNet-18 | 11.7M | BasicBlock × 4 stages | Use this in class — fast, lightweight, easy to experiment with |
| ResNet-34 | 21.8M | BasicBlock × 4 stages (deeper) | Stronger than 18 without the bottleneck overhead |
| ResNet-50 | 25.6M | Bottleneck × 4 stages | **The standard industry baseline** — start here for real projects |
| ResNet-101 | 44.5M | Bottleneck (deeper) | More capacity when ResNet-50 is not enough |
| ResNet-152 | 60.2M | Bottleneck (deepest) | High performance ceiling — only if compute allows |

> **ResNet-50 uses a bottleneck block**: 1×1 Conv (reduce channels) → 3×3 Conv → 1×1 Conv (restore channels). This keeps parameter count manageable while maintaining model capacity.

---

## 6. VGG vs ResNet — Which One and When

| Aspect | VGG | ResNet |
|---|---|---|
| Core idea | Depth by stacking identical conv blocks | Depth with residual skip connections |
| Structure | Sequential, easy to trace | Branching — skip path at every block |
| Teaching clarity | ⭐⭐⭐⭐⭐ Very easy to follow | ⭐⭐⭐ Needs explanation of skip connections |
| Max depth | ~19 layers (VGG-19) | 152+ layers reliably |
| Parameters | ~138M (VGG-16) | ~25.6M (ResNet-50) |
| Memory efficiency | Lower | Better |
| Modern practical use | Teaching, legacy baselines | ✅ Default backbone for new projects |
| Transfer learning | Good | Excellent |

**The one-line version:** Use VGG to understand what a CNN looks like. Use ResNet when you actually want to build something.

---

## 7. The Full Workflow — What You Actually Do in a Project

This is the sequence that works. Follow it in order.

### Step 1 — Understand your data before touching any model

Ask these questions first:
- How many classes?
- How many images per class? Are they balanced?
- Is the domain close to natural photographs or very specialized (medical scans, thermal images)?
- Are labels clean or noisy?

Your answers here will decide how aggressively you can fine-tune later.

### Step 2 — Create clean, separate splits

Split into training, validation, and test sets **before anything else**.

- **Validation set** — used to make decisions during training (learning rate, when to stop, whether to unfreeze)
- **Test set** — locked away, only used once at the very end

If you use the test set to make any training decisions, your final accuracy number is meaningless. It becomes a number that describes how well you overfitted to the test set.

### Step 3 — Normalize using the pretrained model's statistics

If you are using an ImageNet-pretrained model, you must normalize your inputs using ImageNet's exact statistics:

```python
mean = [0.485, 0.456, 0.406]   # per channel (R, G, B)
std  = [0.229, 0.224, 0.225]
```

The pretrained model was trained on data in this distribution. If you feed it unnormalized images, the activations throughout the network will be off and your results will be worse than they should be — with no obvious error message to tell you why.

### Step 4 — Augmentation on training data only

**Training data:** Random crops, horizontal flips, colour jitter, mild rotation. This helps the model generalize.

**Validation and test data:** No random transforms. Only deterministic resizing and normalization.

This distinction matters. If you apply random augmentation to validation data, your validation accuracy will fluctuate every epoch depending on which random crops were generated. You cannot compare results reliably or know when the model has actually improved.

### Step 5 — Load the pretrained backbone and replace the head

Load ResNet or VGG with pretrained weights. Freeze the entire backbone. Then replace only the final layer — because that layer outputs 1000 ImageNet classes and your problem has a different number.

```python
# Load pretrained ResNet-18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze everything
for param in model.parameters():
    param.requires_grad = False

# Replace the head for your task (e.g. 5 classes)
model.fc = nn.Linear(model.fc.in_features, 5)
# model.fc is newly created → it trains; the rest stays frozen
```

### Step 6 — Train the head and record your baseline

Run training with only the new head updating. This is fast because almost nothing is changing. What you get is your **baseline** — the best this frozen backbone can do on your task.

Write that number down. It is your reference point for every decision that follows.

### Step 7 — Fine-tune deeper layers only if the baseline is not enough

If your validation accuracy has plateaued and is not meeting requirements, unfreeze the last one or two residual stages and continue training with smaller learning rates:

```python
# Unfreeze last residual stage
for param in model.layer4.parameters():
    param.requires_grad = True

# Discriminative learning rates
optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), "lr": 1e-5},  # small — protect pretrained features
    {"params": model.fc.parameters(),     "lr": 1e-4},  # larger — head still learning fast
], weight_decay=1e-4)
```

Do not jump straight to unfreezing the entire backbone. Unfreeze one stage, validate, decide whether to go further.

### Step 8 — Evaluate properly

Check accuracy, but also inspect:
- Confusion matrix
- Precision and recall per class
- F1 score
- Actual wrong predictions — look at them with your eyes

A model that gets 92% overall but completely fails on one class is not a good model. The confusion matrix will show you this; accuracy alone will hide it.

---

## 8. PyTorch Code — The Patterns You Will Use

### ResNet-18 — Feature Extraction (Head Only)

```python
import torch
import torch.nn as nn
from torchvision import models

num_classes = 5

# 1. Load pretrained ResNet-18 (ImageNet weights)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 2. Freeze ALL pretrained parameters
for param in model.parameters():
    param.requires_grad = False

# 3. Replace only the final classification layer
#    in_features = 512 for ResNet-18
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
# model.fc was just created → requires_grad=True by default → it will train

# 4. Only pass the new head parameters to the optimizer
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
```

**Why this works:** When you assign a new `nn.Linear` to `model.fc`, it is a freshly initialized module. Its parameters have `requires_grad=True` by default, so they update even though everything else is frozen.

---

### ResNet-18 — Partial Fine-Tuning (layer4 + Head)

```python
# After head-only baseline has plateaued, unfreeze last residual stage
for param in model.layer4.parameters():
    param.requires_grad = True

# Discriminative learning rates:
# - layer4 gets a much smaller LR (protect pretrained features)
# - fc head gets a larger LR (it is still adapting to the task)
optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), "lr": 1e-5},
    {"params": model.fc.parameters(),     "lr": 1e-4},
], weight_decay=1e-4)
```

> **Why different learning rates?** The pretrained layers already contain useful knowledge built from 1.2 million images. A large learning rate would overwrite that knowledge with noise from your small dataset. A small rate nudges them gently toward your domain without destroying what they already know.

---

### VGG-16 — Head Replacement

```python
model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

# VGG has two parts: model.features (convolution backbone)
# and model.classifier (the three FC layers)

# Freeze the convolution backbone
for param in model.features.parameters():
    param.requires_grad = False

# Replace only the last FC layer (index 6) in the classifier
in_features = model.classifier[6].in_features   # 4096
model.classifier[6] = nn.Linear(in_features, num_classes)

# Train the entire classifier block (head is newly replaced)
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
```

The logic is identical to ResNet — freeze the backbone, replace the final output layer, train the head. The only difference is navigating VGG's named structure (`features` vs `classifier`).

---

### Data Pipeline — Normalization Is Not Optional

```python
from torchvision import transforms

# Training: add randomness to help generalization
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),       # random crop
    transforms.RandomHorizontalFlip(),        # random flip
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],           # ImageNet mean
        std =[0.229, 0.224, 0.225]            # ImageNet std
    ),
])

# Validation: deterministic ONLY — no randomness
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),            # fixed size
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    ),
])
```

Notice what is different: the validation pipeline has no random crop and no random flip. If you add randomness to validation, your validation metrics will vary each epoch based on which random transforms were applied — not based on whether your model actually improved.

---

### BatchNorm During Fine-Tuning — Advanced Note

If your batch size is very small during fine-tuning (8 or fewer), BatchNorm statistics become unreliable because they are computed from too few samples. One fix is to keep BatchNorm layers in eval mode:

```python
def set_bn_eval(module):
    if isinstance(module, nn.BatchNorm2d):
        module.eval()

model.apply(set_bn_eval)
```

This makes BatchNorm use its stored running statistics (from pretraining) instead of computing new statistics from your tiny batch. This is not a rule you always apply — it is a debugging tool for when small-batch fine-tuning behaves erratically.

---

## 9. Domain Gap and Catastrophic Forgetting

### Domain Gap

Transfer learning does not work equally well in every situation. It works best when the source domain (ImageNet) and your target domain are visually close. The distance between them is called the **domain gap**.

```
Small Gap ──────────────────────────────── Large Gap

ImageNet      ImageNet       ImageNet       ImageNet
→ Dogs/Cats   → Plant        → Chest         → Thermal
              Disease        X-Rays          Satellite

   ↑               ↑              ↑               ↑
Freeze most   Unfreeze      Unfreeze more    Fine-tune
layers        last block    + domain aug     aggressively
```

A larger gap does not mean transfer learning fails — it means:
- You may need to unfreeze more layers
- You may need domain-specific augmentation
- You need to validate more carefully

### Catastrophic Forgetting

This is what happens when you use a learning rate that is too large during fine-tuning.

The gradient updates are so large that they overwrite the useful patterns the pretrained model spent 1.2 million images learning. The model "forgets" everything and is now just a poorly initialized model being trained from effectively random weights on your small dataset.

**This is why fine-tuning uses small learning rates** — not as a convention, but as a direct consequence of respecting the value in those weights. Use 1e-5 or smaller for any pretrained layer you unfreeze.

---

## 10. Mistakes to Avoid

These are not edge cases. They are extremely common, and most of them are **silent** — you will not get an error. The model will just train badly and you will not know why.

### Mistake 1 — Unfreezing the entire backbone from epoch one

You start with a small dataset, unfreeze everything, and begin training. What happens: the model overfits immediately, and the large gradients from your noisy small dataset overwrite the pretrained features. You end up with something worse than the frozen baseline.

**Fix:** Always start frozen. Get a baseline first.

---

### Mistake 2 — Using a high learning rate on pretrained layers

Pretrained weights are not random — they represent knowledge from 1.2 million images. A high learning rate treats them as if they are random and overwrites them in the first few epochs.

**Fix:** Use 1e-5 or lower for any pretrained layer you unfreeze. Use discriminative learning rates.

---

### Mistake 3 — Forgetting ImageNet normalization

The pretrained model was trained on images normalized to a specific mean and std. If you feed it raw (0–255) or 0–1 normalized images instead of ImageNet-normalized images, the activations throughout the network will be shifted from what the model expects. Your code runs, there is no error, but your results are worse than they should be.

**Fix:** Always apply `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]` when using any ImageNet-pretrained model.

---

### Mistake 4 — Applying random augmentation to validation data

If your validation images are randomly cropped or flipped each epoch, validation accuracy will jump around for reasons unrelated to model quality. You cannot tell whether your model improved or whether it just got lucky with the augmentation draw.

**Fix:** Validation and test pipelines should have zero randomness — only deterministic resize and normalization.

---

### Mistake 5 — Reporting only overall accuracy

A model can get 90% overall accuracy while getting 20% accuracy on one specific class that has fewer samples. The majority classes are pulling the average up and hiding the failure.

**Fix:** Always report per-class metrics and inspect the confusion matrix.

---

### Mistake 6 — Not looking at wrong predictions

You see a low score but have no idea why. Just looking at 20–30 actual wrong predictions almost always tells you: is this a model problem, a data quality problem, a label error, or class ambiguity?

**Fix:** After every evaluation, look at the actual misclassified examples. It is the fastest path to diagnosis.

---

### Mistake 7 — Replacing the wrong layer or the wrong way

If you accidentally replace the wrong index, or forget to check that the new layer's output size matches your number of classes, the model will either throw a shape error at training time or silently produce wrong results.

**Fix:** Print `model` after replacement and verify the architecture. Confirm `model.fc.out_features == num_classes` (for ResNet) before starting training.

---

## 11. Evaluation — Beyond Accuracy

### Why Accuracy Alone Is Not Enough

Suppose you have 5 classes and overall validation accuracy is 88%. That sounds reasonable. But what if class 3 is being predicted correctly only 20% of the time? The other four classes are pulling the average up and hiding that failure.

If you only report accuracy, you will ship a model that works well for 4 classes and fails completely on the fifth.

### The Metrics You Should Always Check

**Accuracy** — overall fraction correct. Useful as a summary, misleading when classes are imbalanced. Never report this alone.

**Precision** — of everything the model predicted as class X, how many actually were class X? High precision = few false positives. Matters when acting on a prediction is costly.

**Recall** — of everything that actually is class X, how many did the model catch? High recall = few false negatives. Matters when missing a case is costly (disease detection, fraud, safety).

**F1 Score** — the harmonic mean of precision and recall:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Use F1 as your primary metric when classes are imbalanced. It balances both failure modes.

**Confusion Matrix** — a table where rows are true labels and columns are predicted labels. Every off-diagonal entry is an error. Read it class by class. This tells you not just how much the model is failing, but *which classes are being confused with which* — which is the actually useful information.

**Error Analysis** — go look at actual wrong predictions. Sometimes the answer is obvious: blurry images, wrong labels, ambiguous cases that even a human would struggle with. This is the fastest path to understanding what is actually wrong.

---

### Training Modes Compared

| Mode | Data You Need | Speed | Risk | When to Use |
|---|---|---|---|---|
| From scratch | Very large — hundreds of thousands | Slow | High — likely to overfit on small data | Only when ImageNet features genuinely don't transfer |
| Feature extraction | Low to medium — even a few thousand works | Fast | Low — backbone is frozen | Your first step. Always. |
| Fine-tuning | Medium to large — more data justifies more unfreezing | Medium | Medium — needs small LR and careful validation | After your baseline plateaus and you have enough data |

---
