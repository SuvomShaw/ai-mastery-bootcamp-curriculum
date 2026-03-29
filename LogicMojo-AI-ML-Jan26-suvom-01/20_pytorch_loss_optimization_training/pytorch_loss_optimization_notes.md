# Training in PyTorch: Loss Functions, Optimizers & Training Loops

---

## What you should be able to do after this lesson

By the end, you should be able to:

- Explain what it means to “train” a model (model + loss + optimizer)
- Choose the right loss function for regression vs classification
- Use **CrossEntropyLoss** and **BCEWithLogitsLoss** correctly (and avoid common mistakes)
- Understand the role of **learning rate** and why it’s the most important knob
- Use modern optimizers (**AdamW**) and basic schedulers
- Write a clean training loop and diagnose issues from loss curves

---

## Prerequisites

You should be comfortable with:

- PyTorch tensors (shape, dtype, device)
- `nn.Module` (`__init__`, `forward`)
- Forward pass → loss → backward pass → optimizer step
- Basic intuition for derivatives (“slope”)

---

## Key definitions (quick glossary)

Use this as a reference while reading the rest of the notes.

- **Parameter ($\theta$)**: A trainable number inside the model (weights and biases). Training changes these.
- **Prediction ($\hat{y}$, “y-hat”)**: The model’s output for an input $x$.
- **Target ($y$)**: The ground-truth answer from the dataset.
- **Loss function ($\ell(\hat{y}, y)$)**: A formula that measures how wrong a prediction is.
- **Loss value ($L$)**: The single scalar number you get after applying the loss function to a *batch*.
- **Gradient ($\nabla_{\theta}L$)**: The derivative of the loss w.r.t. the parameters; tells how $L$ changes if $\theta$ changes.
- **Learning rate ($\alpha$ or `lr`)**: Step size used when updating parameters.
- **Optimizer**: The algorithm that updates parameters using gradients (e.g., SGD, Adam, AdamW).
- **Epoch**: One full pass over the entire training dataset.
- **Batch / mini-batch**: A small subset of training samples processed together in one forward/backward step.
- **Logits**: Raw model outputs *before* applying softmax/sigmoid. They can be any real numbers.
- **Softmax**: Converts a vector of logits into probabilities that sum to 1 (used for multi-class).
- **Sigmoid**: Converts one logit into a probability in $(0, 1)$ (used for binary / multi-label).
- **Weight decay** (`weight_decay`): A regularization method that gently pushes weights toward 0 to reduce overfitting.
- **Momentum**: A technique (mainly in SGD) that uses a running average of past gradients to smooth updates.
- **Scheduler**: A rule that changes the learning rate during training (e.g., step down, cosine decay).
- **Overfitting**: Training loss improves but validation loss gets worse.
- **Underfitting**: Both training and validation performance are poor (model is too simple or not trained enough).
- **Gradient clipping**: Limits gradient size to avoid exploding gradients.

---

## 1) The big picture: what training actually is

Training = finding parameters $\theta$ that make predictions match the targets.

- **Model**: $\hat{y} = f(x; \theta)$
- **Loss**: how wrong $\hat{y}$ is compared to $y$
- **Optimizer**: how we update $\theta$ to reduce the loss

Mathematically:

$$
\theta^* = \arg\min_{\theta} \sum_i \ell\big(f(x_i; \theta), y_i\big)
$$

**What this equation is saying:**

- $x_i$ is the $i$-th input example, and $y_i$ is its true label/target.
- $f(x_i; \theta)$ is the model’s prediction using parameters $\theta$.
- $\ell(\cdot)$ measures “how wrong” the prediction is for one example.
- The sum $\sum_i$ adds up the error over the dataset.
- $\arg\min_{\theta}$ means: “choose the parameters $\theta$ that make this total error as small as possible.”

**Mini example (what you’re minimizing):**

Suppose you have 3 training samples. For some current parameters $\theta$ your model makes predictions and gets per-sample losses:

- sample 1: $\ell_1 = 0.7$
- sample 2: $\ell_2 = 0.2$
- sample 3: $\ell_3 = 1.1$

Total loss (sum) $= 0.7 + 0.2 + 1.1 = 2.0$.

Training tries different $\theta$ values (using gradients) to make this total smaller.

If your Markdown viewer doesn’t render math, read it as:

```text
θ* = argmin_θ  Σ_i  ℓ(f(x_i; θ), y_i)
```

### Loss notation (use this throughout)

You will see two symbols used for “loss”:

- **$\ell(\hat{y}, y)$**: the *loss function* applied to predictions and targets (think: “per-example formula”).
- **$L$**: the *scalar loss value* you get for a batch in code.

In most PyTorch training loops, `loss_fn(...)` returns a **single scalar** (because it reduces across the batch using `mean` by default). That scalar is what we call $L$:

$$
L = \ell(\hat{y}, y)
$$

**Meaning:**

- $\hat{y}$ (“y-hat”) = model prediction for the current batch.
- $y$ = true targets for the current batch.
- $L$ = a single number (scalar) that tells you how wrong the batch predictions are.

**Mini example (batch loss as one number):**

If your batch has $B=4$ samples and their losses are `[0.2, 0.5, 0.1, 0.4]`, then with mean-reduction:

- $L = (0.2 + 0.5 + 0.1 + 0.4)/4 = 0.3$

That `0.3` is what you see as `loss.item()`.

If you want to be more explicit about the batch reduction (batch size $B$):

$$
L = \frac{1}{B}\sum_{i=1}^{B} \ell\big(\hat{y}_i, y_i\big)
$$

**Meaning:**

- $B$ is the batch size.
- $\ell(\hat{y}_i, y_i)$ is the loss for the $i$-th example inside the batch.
- The $\frac{1}{B}$ makes it an **average** loss per example (this matches PyTorch’s default `reduction="mean"`).

**Mini example (same as above, written as a formula):**

For $B=4$ and losses $\ell_1=0.2,\, \ell_2=0.5,\, \ell_3=0.1,\, \ell_4=0.4$:

$$
L = \frac{1}{4}(0.2 + 0.5 + 0.1 + 0.4) = 0.3
$$

And the “full dataset” objective (dataset size $N$) is:

$$
\mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^{N} \ell\big(f(x_i; \theta), y_i\big)
$$

**Meaning:**

- $N$ is the total number of examples in the dataset.
- $\mathcal{L}(\theta)$ is the “overall training objective” we want to minimize.
- In practice, we don’t compute this full sum every step; we approximate it using mini-batches.

**Mini example (dataset-average loss):**

If $N=3$ samples have losses $[0.7, 0.2, 1.1]$, then:

$$
\mathcal{L}(\theta) = (0.7 + 0.2 + 1.1)/3 \approx 0.667
$$

### The training loop (conceptual)

For each epoch and batch:

1. Forward: $\hat{y} = f(x; \theta)$
2. Loss (scalar): $L = \ell(\hat{y}, y)$
3. Backward: compute gradients $\nabla_{\theta} L$
4. Update: $\theta \leftarrow \theta - \alpha \nabla_{\theta} L$

---

## 2) Loss functions

A loss function defines what “wrong” means. The model will get very good at minimizing whatever you choose—so choose carefully.

Throughout this section, $\ell(\hat{y}, y)$ is the loss formula, and $L$ is the scalar loss value returned by `loss_fn(...)` for a batch.

### 2.1 Regression losses (continuous targets)

#### Mean Squared Error (MSE) — `nn.MSELoss()`

- Use when: regression with *mostly clean* data (outliers are rare)

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

**Meaning:**

- $y_i$ is the true value, $\hat{y}_i$ is the predicted value.
- We square the error so negative/positive errors don’t cancel, and big mistakes get punished more.
- $\frac{1}{N}$ averages over $N$ values.

**Mini example:**

True values $y=[2, 4]$, predictions $\hat{y}=[3, 1]$.

- errors: $[-1, 3]$
- squared errors: $[1, 9]$
- MSE: $(1+9)/2 = 5$

### MSE 

**What it does :**

MSE measures “How far are my predictions from the true values?” by:

1. taking the difference (error) for each example,
2. squaring it (so negatives don’t cancel positives, and big mistakes hurt more),
3. averaging those squared errors.

**Why squaring is useful:**

- It makes the loss **smooth** (easy for gradient-based learning).
- It punishes large mistakes a lot. If you’re off by 10, that contributes $10^2=100$.

**When to use MSE:**

- Regression problems where **large errors should be punished strongly**.
- Data where outliers are rare (typical “nice” noise).

**When NOT to use MSE:**

- When your dataset has outliers or occasional corrupted labels.
  - One extreme outlier can dominate the loss and “pull” the model in the wrong direction.

**How to interpret MSE values:**

- MSE is in **squared units**.
  - If $y$ is in “rupees”, MSE is in “rupees²”.
- Many people track **RMSE** ($\sqrt{\text{MSE}}$) because it goes back to the original units.

**Common practical tips:**

- Scale/normalize your targets if values are very large (helps stable training).
- Keep `reduction="mean"` (default) so the loss is not dependent on batch size.

Why it’s popular:
- Big errors are punished more (squared)
- Smooth gradients

Downside:
- Outliers can dominate training

```python
import torch              # Core PyTorch package
import torch.nn as nn     # Neural-network layers and loss functions

loss_fn = nn.MSELoss()    # Mean Squared Error loss

y_pred = torch.tensor([3.0, 5.0, 2.5])  # Model predictions (ŷ)
y_true = torch.tensor([2.5, 5.0, 3.0])  # True targets (y)

loss = loss_fn(y_pred, y_true)          # Compute scalar loss value L
print(loss.item())                      # Convert tensor to Python float for printing
```

#### Mean Absolute Error (MAE) — `nn.L1Loss()`

- Use when: regression with **outliers**

$$
\mathcal{L}_{\text{MAE}} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|
$$

**Meaning:**

- Same as MSE, but we take absolute error instead of squaring.
- Errors grow **linearly**, so outliers don’t dominate as much as with MSE.

**Mini example (same numbers as above):**

True values $y=[2, 4]$, predictions $\hat{y}=[3, 1]$.

- absolute errors: $[1, 3]$
- MAE: $(1+3)/2 = 2$

### MAE 

**What it does:**

MAE measures the average absolute distance between predictions and true values.
Think of it as: “On average, how many units am I off?”

**Why it’s different from MSE:**

- MAE grows **linearly** with the error.
  - Being off by 10 is exactly 10× worse than being off by 1.
- That makes MAE **robust to outliers**.

**When to use MAE:**

- Regression with outliers (real-world noisy measurements).
- When you want a loss that matches “average absolute mistake”.

**Trade-off:**

- MAE is less smooth than MSE (the absolute value has a sharp corner at 0).
  - In practice, it can converge a bit slower or feel less “stable” for some models.

**How to interpret MAE values:**

- MAE is in the **same units** as your target.
  - If temperature is in °C, MAE is also in °C.

Notes:
- Less sensitive to outliers
- Gradient is not smooth at 0 (can converge a bit differently than MSE)

```python
loss_fn = nn.L1Loss()                   # Mean Absolute Error (L1) loss
loss = loss_fn(y_pred, y_true)          # Compute scalar loss value L
print(loss.item())                      # Print as a Python float
```

#### Huber loss — `nn.HuberLoss()`

- Use when: you want a strong default for regression when you’re unsure
- Acts like MSE for small errors and like MAE for big errors

**Mini example (intuition):**

- For small errors like $0.2$, Huber behaves like MSE: it punishes smoothly.
- For big errors like $10$, Huber behaves closer to MAE: it avoids the huge squaring effect.

That’s why it’s often a safer default when your data might contain outliers.

### Huber (the “best of both”)

**What it does:**

Huber loss behaves like:

- **MSE for small errors** (smooth learning when you’re already close), and
- **MAE for big errors** (doesn’t let outliers explode the loss).

**Why people like it:**

If you’re unsure whether your data has outliers, Huber is a strong default because it usually trains smoothly like MSE, but stays safer like MAE.

**The Math:**

$$\mathcal{L}_{\delta}(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\ \delta \cdot (|y - \hat{y}| - \frac{\delta}{2}) & \text{otherwise} \end{cases}$$

**Key knob: `delta`**

- `delta` decides when the loss switches from “square” behavior to “absolute” behavior.
- Smaller `delta` → becomes MAE-like sooner.
- Larger `delta` → behaves more like MSE for a wider range of errors.

```python
loss_fn = nn.HuberLoss(delta=1.0)       # Huber loss; delta controls the switch point
loss = loss_fn(y_pred, y_true)          # Compute scalar loss value L
print(loss.item())                      # Print as a Python float
```

---

### 2.2 Classification losses

#### Cross-Entropy (multi-class) — `nn.CrossEntropyLoss()`

- Use when: **multi-class** classification (classes are mutually exclusive)
- Input: **raw logits** of shape `(batch_size, num_classes)`
- Target: class indices of shape `(batch_size,)` with dtype `torch.long`

**Definitions (important):**

- **Class index**: An integer label like `0, 1, 2, ... K-1`.
- **Logits**: raw scores from the model (no softmax). PyTorch expects logits here.

Internally, PyTorch computes:

- `LogSoftmax(logits)`
- then negative log-likelihood

So you should **not** apply softmax yourself.

### Cross-entropy 

**What it measures:**

Cross-entropy answers: “How confident was the model about the correct class?”

It becomes small when the model assigns **high probability** to the true class, and large when it assigns **low probability**.

If the predicted probability of the true class is $p_{true}$, the loss is roughly:

$$
L \approx -\log(p_{true})
$$

**How to interpret the value:**

- If $p_{true}=1.0$ → $L = -\log(1)=0$ (perfect confidence).
- If $p_{true}=0.5$ → $L \approx 0.693$.
- If $p_{true}=0.1$ → $L \approx 2.303$ (confidently wrong / very unsure).

**Why logits are used (and not softmax output):**

- Softmax can cause numerical issues for large/small numbers.
- PyTorch’s `CrossEntropyLoss` combines `LogSoftmax + NLLLoss` internally in a stable way.
- If you apply `softmax` yourself, you typically make training less stable.

**When to use cross-entropy:**

- Any time exactly **one** class is correct out of $K$ (cat vs dog vs bird).

**Mini example (logits → prediction):**

If your model outputs logits `[2.0, 1.0, 0.1]` for 3 classes, the largest logit is `2.0` (class 0), so `argmax` predicts class `0`.

During training, `CrossEntropyLoss` uses these logits to compute a probability distribution internally (via log-softmax) and penalizes the model if the true class is not class 0.

**Common mistake (very important):**

- ✅ Correct: pass logits into `CrossEntropyLoss`
- ❌ Wrong: apply `softmax` first, then `CrossEntropyLoss` (double-softmax, worse numerics)

```python
import torch
import torch.nn as nn

loss_fn = nn.CrossEntropyLoss()         # Multi-class classification loss

logits = torch.tensor([[2.0, 1.0, 0.1]])   # Raw scores (logits), shape (batch=1, classes=3)
target = torch.tensor([0])                 # True class index, shape (batch=1,), dtype long

loss = loss_fn(logits, target)          # Compute scalar loss value L
print(loss.item())                      # Print as a Python float
```

**Quick sanity check:**
For a randomly initialized model with $K$ classes, initial loss is often close to:

$$
\log(K)
$$

**Meaning (why this is useful):**

- If the model is guessing randomly among $K$ classes, it assigns about $1/K$ probability to the correct class.
- Cross-entropy becomes approximately $-\log(1/K) = \log(K)$.
- So $\log(K)$ is a quick “sanity check” for your very first loss value.

**Mini example:**

If you have $K=5$ classes, then $\log(5) \approx 1.609$.

So when you start training a randomly initialized 5-class model, seeing an initial loss near ~1.6 is normal.

Example: for 10 classes, $\log(10) \approx 2.3026$.

---

#### Binary Cross-Entropy with logits — `nn.BCEWithLogitsLoss()`

- Use when:
  - Binary classification with **one output neuron**, or
  - Multi-label classification (each class is independent)

- Input: raw logits (no sigmoid in the model)
- Target: float labels (0.0/1.0)

**Definitions (important):**

- **Binary**: exactly 2 classes (0 or 1).
- **Multi-label**: a sample can belong to multiple classes at once (each label is an independent 0/1 decision).
- For BCE, targets are floats (`0.0` or `1.0`) because it treats labels as probabilities.

**Mini example (binary):**

- If a logit is `+4`, sigmoid(logit) is close to 1 → model is very confident for class 1.
- If a logit is `-4`, sigmoid(logit) is close to 0 → model is very confident for class 0.

`BCEWithLogitsLoss` combines sigmoid + BCE in one stable function, so you pass the raw logits.

### BCEWithLogits  (binary + multi-label)

**What it measures:**

For each output logit, the model is trying to answer a yes/no question:

- “Is this class present?” (1) or “not present?” (0)

It penalizes confident wrong answers heavily, and rewards confident correct answers.

**Binary vs multi-label:**

- **Binary classification (single output):** one yes/no decision.
- **Multi-label (K outputs):** K independent yes/no decisions.
  - Example: an image can be *both* “outdoor” and “contains people”.

**Why this loss uses sigmoid:**

- Sigmoid turns each logit into an independent probability between 0 and 1.
- Unlike softmax, probabilities do **not** have to sum to 1.

**Common mistakes:**

- Using `nn.Sigmoid()` in the model and then using `BCEWithLogitsLoss` (double-sigmoid).
- Using integer targets (`LongTensor`) instead of float targets.

```python
import torch
import torch.nn as nn

loss_fn = nn.BCEWithLogitsLoss()        # Binary / multi-label loss (sigmoid is inside)

# Binary (1 logit)
logit  = torch.tensor([1.5])            # Raw score for class 1 (positive => predicts 1)
target = torch.tensor([1.0])            # True label as float (0.0 or 1.0)
loss = loss_fn(logit, target)           # Compute scalar loss value L
print(loss.item())                      # Print as a Python float

# Multi-label (K logits)
logits = torch.tensor([[1.5, -0.5, 2.0]]) # 3 independent binary logits, shape (B=1, K=3)
target = torch.tensor([[1.0,  0.0, 1.0]]) # 3 independent binary labels (floats), same shape
loss = loss_fn(logits, target)            # Compute scalar loss value L
print(loss.item())                        # Print as a Python float
```

#### Which one should you use?

**What does `Linear(..., K)` mean?**

- `Linear(in_features, out_features)` is the last layer that maps the model’s hidden representation to output scores.
- The `...` means “whatever number of input features your network has at that point” (often written as `in_features`).
- `K` is the number of classes.

So `Linear(..., K)` means: “the final layer produces **K logits**, one per class.”

Example: if your model’s last hidden layer has 128 features and you have 10 classes, the final layer is:

```python
nn.Linear(128, 10)  # 10 logits => 10-class classification
```

| Problem type | Output layer | Loss | Target dtype/shape |
|---|---|---|---|
| Multi-class (K classes, one correct) | `Linear(..., K)` | `CrossEntropyLoss` | `LongTensor`, `(B,)` |
| Binary, 1 output neuron | `Linear(..., 1)` | `BCEWithLogitsLoss` | `FloatTensor`, `(B, 1)` or `(B,)` |
| Multi-label (K independent labels) | `Linear(..., K)` | `BCEWithLogitsLoss` | `FloatTensor`, `(B, K)` |

---

## 3) Gradient descent (the basic update)

The core idea:

$$
\theta \leftarrow \theta - \alpha \nabla_{\theta} L
$$

**Meaning:**

- $\theta$ = all trainable parameters (weights and biases).
- $\nabla_{\theta} L$ = gradient of the loss w.r.t. the parameters (tells you which direction increases the loss fastest).
- $\alpha$ = learning rate (how big a step to take).
- The minus sign means we move **downhill** to reduce the loss.

**Mini example (one parameter):**

Assume a single parameter $\theta=2.0$, learning rate $\alpha=0.1$, and gradient $\nabla_{\theta}L = 3.0$.

$$
\theta_{new} = 2.0 - 0.1\times 3.0 = 1.7
$$

So the optimizer moves $\theta$ a bit in the direction that reduces the loss.

In practice, we want to minimize the dataset objective $\mathcal{L}(\theta)$, but we estimate its gradient using mini-batches (so we update using $\nabla_{\theta} L$ from the current batch).

### Mini-batch gradient descent

In deep learning, you almost always train using **mini-batches**.

- Full batch: accurate direction, slow updates
- Stochastic (1 sample): very noisy
- Mini-batch: good balance, noise can actually help optimization

### Exploding gradients and clipping

If gradients become too large, loss can become `inf` or `nan`.

```python
# after loss.backward()                  # Gradients must exist before clipping
max_norm = 1.0                           # Maximum allowed L2 norm for the gradients
torch.nn.utils.clip_grad_norm_(           # Clip gradients in-place
    model.parameters(),                   # Parameters whose gradients will be clipped
    max_norm                              # Threshold
)
```

---

## 4) Optimizers (how updates are chosen)

Think of training like walking down a hill in thick fog.

- The **loss** is your altitude.
- The **gradient** tells you the steepest downhill direction.
- The **optimizer** is your walking strategy.

Different optimizers answer questions like:

- “Should I take small steps or large steps?”
- “If the slope keeps pointing the same way, should I build up speed?”
- “If some parameters get huge gradients and others get tiny gradients, should they all use the same step size?”

### 4.1 SGD with momentum

Momentum helps reduce zig-zagging and speeds progress in consistent directions.

**Definition:** Momentum is like “inertia” for parameter updates: if gradients keep pointing in the same direction, updates build up speed.

### SGD + Momentum 

**What SGD does:**

SGD updates parameters using the current batch gradient. It’s the most basic “take a step downhill” method.

**Why momentum helps:**

Without momentum, SGD can:

- zig-zag in narrow valleys,
- get slowed down by noisy mini-batches.

Momentum acts like rolling a heavy ball:

- if the slope keeps pointing the same way, the ball accelerates,
- if one batch is noisy, the ball doesn’t instantly change direction.

**When to use:**

- Classic choice for **computer vision** (CNNs) when you can tune LR well.
- Often generalizes well, but may need more careful LR scheduling.

**How to tune (rule of thumb):**

- Start with `momentum=0.9`.
- LR typically larger than Adam/AdamW (e.g., 0.1 → 0.01 range depending on task).

```python
optimizer = torch.optim.SGD(              # Stochastic Gradient Descent optimizer
    model.parameters(),                   # Parameters to optimize
    lr=0.01,                              # Learning rate
    momentum=0.9                          # Momentum coefficient
)
```

### 4.2 Adam

Adam adapts the step size per parameter and uses momentum-like running averages.

**Definition:** Adam keeps running estimates of (1) average gradient direction and (2) how large/variable the gradients are, then uses them to pick an effective step size per parameter.

### Adam 

**What Adam does:**

Adam is like SGD + momentum + “automatic per-parameter step sizes”.

Some weights get big gradients often, others rarely get updated.
Adam adapts so that:

- parameters with consistently large gradients don’t explode (smaller effective steps),
- parameters with tiny/rare gradients still learn (larger effective steps when needed).

**Why beginners like Adam:**

- Works well out-of-the-box on many problems.
- Less LR tuning compared to SGD.

**When to use:**

- As a fast baseline.
- For many NLP / tabular / general problems.

**Common pitfall:**

- Adding L2 regularization incorrectly; in modern practice, prefer AdamW when using weight decay.

```python
optimizer = torch.optim.Adam(             # Adam optimizer
    model.parameters(),                   # Parameters to optimize
    lr=1e-3                               # Learning rate
)
```

### 4.3 AdamW (recommended default)

AdamW fixes weight decay handling (decoupled weight decay). This is the modern default for many deep learning tasks.

**Definition:** Weight decay is a regularization term that shrinks weights a little each step to reduce overfitting. AdamW applies this shrinkage in a clean, decoupled way.

### AdamW (why it’s the modern default)

**What AdamW changes compared to Adam:**

AdamW applies weight decay as a separate “shrink weights slightly” step, instead of mixing it into the gradient in a way that interacts badly with Adam’s adaptive scaling.

**Why that matters (simple intuition):**

- Weight decay is supposed to consistently discourage large weights.
- In Adam (old style), adaptive scaling can unintentionally weaken weight decay for some parameters.
- AdamW makes weight decay behave more predictably.

**When to use:**

- Most modern deep learning training (Transformers, ViTs, fine-tuning).
- Whenever you want regularization via `weight_decay`.

**Typical starting point:**

- `lr=1e-3, weight_decay=1e-2` for small models.
- For fine-tuning large pretrained models, `lr` is usually much smaller (e.g., `1e-5` to `5e-5`).

```python
optimizer = torch.optim.AdamW(            # AdamW optimizer (Adam + decoupled weight decay)
    model.parameters(),                   # Parameters to optimize
    lr=1e-3,                              # Learning rate
    weight_decay=1e-2                     # Weight decay (regularization strength)
)
```

### Layer-wise learning rates (common in fine-tuning)

```python
optimizer = torch.optim.AdamW([
    {"params": model.backbone.parameters(), "lr": 1e-5},  # Pretrained layers: small LR
    {"params": model.head.parameters(),     "lr": 1e-3},  # New layers: larger LR
], weight_decay=1e-2)                                          # Same weight decay for all groups
```

---

## 5) Learning rate (the most important knob)

If LR is wrong, everything else becomes hard.

Typical behaviors:

- LR too small → loss decreases extremely slowly
- LR good → smooth, fast decrease
- LR too big → oscillation or divergence (`nan`)

**Mini example (why LR can break training):**

Assume one parameter and gradient $\nabla_{\theta}L = 3$.

- With $\alpha=0.01$: step size is $0.03$ (small, stable)
- With $\alpha=1.0$: step size is $3.0$ (can overshoot and bounce)
- With $\alpha=10.0$: step size is $30.0$ (almost guaranteed to diverge)

Rules of thumb (starting points):

| Task | Optimizer | Typical LR |
|---|---|---|
| Small MLP / tabular | AdamW | `1e-3` to `5e-3` |
| CNN from scratch | SGD+momentum | `1e-1` to `1e-2` |
| Transformer fine-tuning | AdamW | `1e-5` to `5e-5` |

---

## 6) Learning-rate schedulers

Schedulers change LR over time.

**Definition:** A scheduler updates `optimizer.param_groups[...]['lr']` during training so the learning rate follows a plan (for example: start higher, then gradually reduce).

Common options:

```python
import torch.optim as optim                 # Optimizers + schedulers live here

optimizer = optim.AdamW(                    # Choose an optimizer
    model.parameters(),                     # Parameters to optimize
    lr=1e-3                                 # Starting learning rate
)

# Step down every N epochs
scheduler = optim.lr_scheduler.StepLR(      # Multiply LR by gamma every step_size epochs
    optimizer,                              # Optimizer whose LR will be scheduled
    step_size=30,                           # Drop every 30 epochs
    gamma=0.5                               # New LR = old LR * 0.5
)

# Smooth cosine decay
scheduler = optim.lr_scheduler.CosineAnnealingLR(  # Cosine decay from lr -> eta_min
    optimizer,                                      # Optimizer whose LR will be scheduled
    T_max=100,                                      # Number of epochs for a full cosine schedule
    eta_min=1e-6                                    # Minimum LR
)

# Reduce when validation loss stops improving
scheduler = optim.lr_scheduler.ReduceLROnPlateau( # Reduce LR when a metric plateaus
    optimizer,                                     # Optimizer whose LR will be scheduled
    mode='min',                                    # We want to minimize validation loss
    patience=10,                                   # Wait 10 epochs without improvement
    factor=0.1                                     # New LR = old LR * 0.1
)
```

### Where to call `scheduler.step()`

- **Epoch-based schedulers** (StepLR, CosineAnnealingLR): call once per epoch
- **ReduceLROnPlateau**: call once per epoch and pass the validation metric
- **Batch-based schedulers** (OneCycleLR): call after every batch

---

## 7) A clean training loop template

This is a practical template you can adapt. Focus on the **order** of operations.

```python
import torch                                  # Core PyTorch
import torch.nn as nn                         # Neural-network utilities


def train_one_epoch(model, loader, loss_fn, optimizer, device, max_grad_norm=None):
    model.train()                             # Enable training mode (dropout/bn behave accordingly)
    total_loss = 0.0                          # Sum of batch losses (for epoch average)
    correct = 0                               # Count correct predictions (classification)
    n = 0                                     # Total number of samples seen

    for X, y in loader:                       # Iterate over mini-batches
        X = X.to(device)                      # Move inputs to CPU/GPU
        y = y.to(device)                      # Move targets to CPU/GPU

        logits = model(X)                      # Forward pass: compute logits
        loss = loss_fn(logits, y)              # Compute scalar loss L = ℓ(ŷ, y)

        optimizer.zero_grad(set_to_none=True) # Clear old gradients (faster with set_to_none)
        loss.backward()                        # Backward pass: compute gradients

        if max_grad_norm is not None:         # Optional safety for exploding gradients
            torch.nn.utils.clip_grad_norm_(    # Clip gradient norm in-place
                model.parameters(),            # Parameters to clip
                max_grad_norm                  # Threshold
            )

        optimizer.step()                       # Update parameters using the optimizer rule

        bs = X.size(0)                         # Batch size
        total_loss += loss.item() * bs         # Accumulate un-averaged loss for epoch mean
        n += bs                                # Update sample count
        correct += (logits.argmax(dim=1) == y).sum().item()  # Count correct predictions

    return total_loss / n, correct / n         # Return average loss and accuracy


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()                              # Eval mode (dropout off, bn uses running stats)
    total_loss = 0.0                          # Sum of batch losses
    correct = 0                               # Count correct predictions
    n = 0                                     # Total number of samples

    for X, y in loader:                       # Iterate over validation batches
        X = X.to(device)                      # Move inputs to device
        y = y.to(device)                      # Move targets to device

        logits = model(X)                      # Forward pass
        loss = loss_fn(logits, y)              # Compute scalar loss

        bs = X.size(0)                         # Batch size
        total_loss += loss.item() * bs         # Accumulate un-averaged loss
        n += bs                                # Update sample count
        correct += (logits.argmax(dim=1) == y).sum().item()  # Count correct predictions

    return total_loss / n, correct / n         # Return average loss and accuracy


def fit(model, train_loader, val_loader, epochs, device):
    loss_fn = nn.CrossEntropyLoss()            # Choose loss function for multi-class classification
    optimizer = torch.optim.AdamW(             # Choose optimizer
        model.parameters(),                    # Parameters to optimize
        lr=1e-3,                               # Starting learning rate
        weight_decay=1e-2                      # Weight decay regularization
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(  # Choose LR schedule
        optimizer,                               # Optimizer whose LR will change
        T_max=epochs,                            # Total epochs for full cosine schedule
        eta_min=1e-6                             # Minimum LR
    )

    history = {                               # Store metrics for plotting later
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    for epoch in range(1, epochs + 1):        # Training loop over epochs
        train_loss, train_acc = train_one_epoch(  # Train on all batches
            model,                               # Model
            train_loader,                        # Training data
            loss_fn,                             # Loss
            optimizer,                           # Optimizer
            device,                              # CPU/GPU
            max_grad_norm=1.0                    # Gradient clipping threshold
        )
        val_loss, val_acc = evaluate(          # Evaluate on validation set
            model,                              # Model
            val_loader,                         # Validation data
            loss_fn,                            # Loss
            device                              # CPU/GPU
        )

        scheduler.step()                      # Step the scheduler once per epoch

        history["train_loss"].append(train_loss)                 # Save train loss
        history["train_acc"].append(train_acc)                   # Save train accuracy
        history["val_loss"].append(val_loss)                     # Save val loss
        history["val_acc"].append(val_acc)                       # Save val accuracy
        history["lr"].append(optimizer.param_groups[0]["lr"])   # Save current LR

        print(                                # Print progress each epoch
            f"Epoch {epoch:03d}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.1%} | "
            f"val loss {val_loss:.4f} acc {val_acc:.1%} | "
            f"lr {optimizer.param_groups[0]['lr']:.2e}"
        )

    return history                            # Return metrics
```

### Batch-step order (don’t shuffle this)

1. `logits = model(X)`
2. `loss = loss_fn(logits, y)`
3. `optimizer.zero_grad(...)`
4. `loss.backward()`
5. (optional) clip gradients
6. `optimizer.step()`

---

## 8) Diagnosing training from curves

Look at **train loss** vs **val loss**:

- Healthy: both go down; val is slightly higher
- Overfitting: train down, val goes up
- Underfitting: both plateau too high
- LR too high: loss oscillates or becomes `nan`
- LR too low: loss barely moves

### Quick checks when something looks wrong

- Is your loss decreasing at all?
- Are your labels correct and dtype correct (`long` for CE, `float` for BCE)?
- Are you accidentally applying `softmax` before `CrossEntropyLoss`?
- Did you forget `optimizer.zero_grad()`?
- Are you calling `model.eval()` and `torch.no_grad()` during validation?

---

## 9) Self-check questions (no peeking)

1. When should you use MSE vs MAE vs Huber?
2. Why must you pass **logits** (not softmax outputs) to `CrossEntropyLoss`?
3. What should the initial loss be for a random model on $K$ classes?
4. What happens if you forget `optimizer.zero_grad()`?
5. Adam vs AdamW: what’s the difference?
6. When do you clip gradients, and why?
7. Train loss decreases but val loss increases—what’s happening?
8. Why do we log `loss.item()` instead of saving the `loss` tensor?
9. Where does `scheduler.step()` go for an epoch-based scheduler?
10. What LR range is common for Transformer fine-tuning?

---

## One-page cheat sheet

**Loss selection**

- Regression (clean): `MSELoss`
- Regression (outliers / unsure): `HuberLoss`
- Multi-class: `CrossEntropyLoss` (**logits in, long targets**)
- Binary / multi-label: `BCEWithLogitsLoss` (**logits in, float targets**)

**Training step (batch)**

1. forward
2. loss
3. `zero_grad`
4. backward
5. (optional) clip
6. `step`

**Common bugs**

- Softmax before `CrossEntropyLoss`
- Wrong target dtype (`float` vs `long`)
- Forgetting `zero_grad()`
- Forgetting `model.eval()` for validation
- Calling epoch scheduler inside the batch loop
