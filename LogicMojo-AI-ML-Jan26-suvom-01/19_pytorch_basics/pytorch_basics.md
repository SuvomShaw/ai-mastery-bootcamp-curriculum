# PyTorch Fundamentals 

## Learning goals
By the end, you should be able to:

- Create and manipulate tensors (shapes, indexing, broadcasting, GPU)
- Understand how PyTorch computes gradients with **autograd**
- Build models using **`nn.Module`**
- Write the standard **training loop** used in almost every deep learning project

---

## Contents

1. [Why PyTorch?](#1-why-pytorch)
2. [Tensors](#2-tensors)
3. [Autograd](#3-autograd)
4. [`nn.Module` basics](#4-nnmodule-basics)
5. [The training loop](#5-the-training-loop)
6. [Mini Lab: Binary classifier (with DataLoader)](#6-mini-lab-binary-classifier-with-dataloader)
7. [One-page cheat sheet](#7-one-page-cheat-sheet)

---
<img src="https://miro.medium.com/1%2AiSsWB9tQPnY-Wi5qzCEaeg.png" width="600" height="400">

<img src="https://miro.medium.com/1%2AZS0PrJ6wS7Q5D0sz0ULE2w.png" width="600" height="400">
---

# 1) Why PyTorch?

PyTorch is popular because it feels like normal Python + NumPy.

## 1.1 Dynamic computation graphs (define-by-run)

PyTorch builds a computation graph **as your code executes**. This makes it easy to debug (you can print tensors, use `if`, use standard Python debuggers).

```python
import torch

x = torch.tensor(3.0)
w = torch.tensor(2.0, requires_grad=True)

y = w * x
z = y + 5

z.backward()           # compute dz/dw
print(w.grad)          # tensor(3.)
```

## 1.2 The “PyTorch Trinity”

```
Tensors   +   Autograd   +   nn.Module
 data         gradients      model definition
```

---

# 2) Tensors

A **tensor** is a multi-dimensional array (like NumPy), with two big upgrades:

- It can live on **GPU**
- It can be tracked by **autograd**

<img src="https://www.researchgate.net/publication/383280310/figure/fig1/AS%3A11431281272889807%401724300351482/sual-representation-of-Pytorch-tensors-From-left-to-right-we-see-a-scalar-x-R-a.png" width="600" height="400">

## 2.1 Shapes and common conventions

```text
0D scalar: 42                     shape: ()
1D vector: [1,2,3]                shape: (3,)
2D matrix: [[...],[...]]          shape: (rows, cols)
3D: (C, H, W)                     single image (channels-first)
4D: (N, C, H, W)                  batch of images

N=batch, C=channels, H=height, W=width
```

PyTorch in vision is usually **channels-first** (`N, C, H, W`).

## 2.2 Creating tensors

```python
import torch

torch.manual_seed(0)  # reproducibility

scalar = torch.tensor(42)
vector = torch.tensor([1.0, 2.0, 3.0])
matrix = torch.tensor([[1, 2], [3, 4]])

print(vector.shape, vector.dtype)  # torch.Size([3]) torch.float32
print(matrix.dtype)                # torch.int64 (default for ints)

# Factory functions
w = torch.randn(3, 4)              # normal distribution
u = torch.rand(2, 3)               # uniform [0,1)
z = torch.zeros(2, 2)
o = torch.ones(2, 2)
I = torch.eye(3)
a = torch.arange(0, 10, step=2)    # [0,2,4,6,8]
lin = torch.linspace(0, 1, steps=5)  # [0.00,0.25,0.50,0.75,1.00]
```

### Dtypes (very important)

Common dtypes:

- `torch.float32` (most common for neural nets)
- `torch.float16` / `torch.bfloat16` (mixed precision)
- `torch.int64` (common for class labels / indices)

```python
x = torch.tensor([1, 2, 3])
print(x.dtype)       # torch.int64

x = x.float()
print(x.dtype)       # torch.float32
```

## 2.3 NumPy ↔ Torch bridge

`torch.from_numpy` shares memory (no copy). Modifying one can modify the other.

```python
import numpy as np
import torch

np_arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(np_arr)   # shares memory

t[0] = 99.0
print(np_arr)  # [99.  2.  3.]

# independent copy
t2 = torch.from_numpy(np_arr).clone()
```

## 2.4 Indexing and slicing

Indexing looks like NumPy.

```python
import torch

X = torch.tensor([
    [10, 11, 12],
    [20, 21, 22],
    [30, 31, 32]
])

print(X[0])        # first row: tensor([10, 11, 12])
print(X[:, 1])     # second column: tensor([11, 21, 31])
print(X[1:, :2])   # rows 1..end, cols 0..1
```

### Boolean masking

```python
import torch

x = torch.tensor([1, -2, 3, -4, 5])
mask = x > 0
print(mask)     # tensor([ True, False,  True, False,  True])
print(x[mask])  # tensor([1, 3, 5])
```

## 2.5 Broadcasting (why shapes “magically” work)

Broadcasting lets PyTorch automatically expand dimensions.

```python
import torch

X = torch.randn(4, 3)          # (batch=4, features=3)
b = torch.randn(3)             # (3,)

Y = X + b                      # b broadcasts to (4, 3)
print(Y.shape)                 # torch.Size([4, 3])
```

If broadcasting fails, you’ll get shape mismatch errors. In that case, check shapes using `.shape`.

## 2.6 Reshaping and dimension helpers

Useful operations:

- `reshape` / `view` (change shape)
- `unsqueeze` (add dimension)
- `squeeze` (remove size-1 dimension)
- `transpose` / `permute` (reorder dimensions)

```python
import torch

x = torch.arange(12)           # (12,)
grid = x.reshape(3, 4)         # (3,4)

v = torch.randn(5)             # (5,)
v2 = v.unsqueeze(0)            # (1,5)
v3 = v.unsqueeze(1)            # (5,1)

print(v.shape, v2.shape, v3.shape)

img = torch.randn(3, 32, 32)   # (C,H,W)
img_hwc = img.permute(1, 2, 0) # (H,W,C)
print(img_hwc.shape)
```

**`view` vs `reshape`**:

- `view` requires contiguous memory (no copy) and can fail after `permute`
- `reshape` is safer; it copies only if needed

## 2.7 CPU ↔ GPU

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

x = torch.randn(2, 3)
x = x.to(device)

print(device)
print(x.device)
```

When using GPU, **model and tensors must be on the same device**.

---

## Common beginner errors (save these!)

### Error 1 — Shape mismatch

```python
import torch

A = torch.randn(3, 2)
B = torch.randn(4, 2)

# A @ B  # RuntimeError

# Fix: transpose
C = A @ B.T  # (3,2) @ (2,4) -> (3,4)
```
### Error 2 — Device mismatch

```python
import torch

x_cpu = torch.tensor([1.0, 2.0])
x_gpu = torch.tensor([3.0, 4.0]).to("cuda")

# x_cpu + x_gpu  # RuntimeError

z = x_cpu.to("cuda") + x_gpu
```

---

# 3) Autograd

Autograd automatically computes gradients using the chain rule.

<img src="https://substackcdn.com/image/fetch/%24s_%215Dsq%21%2Cf_auto%2Cq_auto%3Agood%2Cfl_progressive%3Asteep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2a85738c-d9d7-42e8-a266-fad9e6425e84_1176x840.png" width="600" height="400">

<img src="https://miro.medium.com/v2/resize%3Afit%3A1200/1%2AW7ZPd1tvyi_cIdpoDdX3DA.png" width="600" height="400">

## 3.1 `requires_grad=True`

- Use `requires_grad=True` for **learnable parameters** (weights)
- Inputs usually do **not** need gradients

```python
import torch

w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)

z = w * x + 5
z.backward()

print(w.grad)  # dz/dw = 3
```

## 3.2 What is tracked?

If a tensor is created from operations involving something with `requires_grad=True`, it will have a `grad_fn`.

```python
import torch

w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)

y = w * x
print(y)           # tensor(..., grad_fn=<MulBackward0>)
print(y.grad_fn)   # tells you which op created it
```

## 3.3 Gradients accumulate (important!)

Calling `.backward()` multiple times **adds** gradients.

```python
import torch

w = torch.tensor(2.0, requires_grad=True)

for _ in range(3):
    loss = w * 3.0
    loss.backward()
    print(w.grad)

# Fix: clear gradients
w = torch.tensor(2.0, requires_grad=True)
for _ in range(3):
    loss = w * 3.0
    w.grad = None
    loss.backward()
    print(w.grad)
```

Why does accumulation exist?

- It enables **gradient accumulation** over several mini-batches before an update.


## 3.4 `torch.no_grad()` vs `.detach()`

```python
import torch

w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)

with torch.no_grad():
    y = w * x
    print(y.grad_fn)  # None

z = w * x
z_detached = z.detach()
```

| Feature | `torch.no_grad()` | `.detach()` |
|---|---|---|
| Applies to | a block of code | a tensor |
| Graph creation | prevented | created, then cut |
| Typical use | evaluation/inference | stop gradient flow |

---

# 4) `nn.Module` basics

An `nn.Module` is the standard way to define a neural network.

It bundles:

- **Learnable parameters** (automatically tracked)
- A **forward pass**

<img src="https://miro.medium.com/1%2AjfEsZoTqJ6W_6tE1wMCdhQ.jpeg" width="600" height="400">

<img src="https://miro.medium.com/1%2AZXAOUqmlyECgfVa81Sr6Ew.png" width="600" height="400">

## 4.1 Minimal model template

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()  # important: registers submodules & parameters
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)
```

If you forget `super().__init__()`, parameters might not be registered → optimizer won’t update anything.

## 4.2 Example: small MLP

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=4, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)

model = MLP()
print(model)
print("params:", sum(p.numel() for p in model.parameters()))

dummy = torch.tensor([[1.0, 2.0]])
print(model(dummy))
```

## 4.3 Inspecting parameters

```python
for name, p in model.named_parameters():
    print(f"{name:20s}", p.shape, p.requires_grad)
```

## 4.4 Moving a model to GPU

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

When you do this, you must also move the data:

```python
X = X.to(device)
y = y.to(device)
```

## 4.5 `train()` vs `eval()`

Some layers change behavior:

- **Dropout** is ON during training, OFF during evaluation
- **BatchNorm** uses batch statistics during training, running averages during evaluation

```python
model.train()  # training mode
model.eval()   # evaluation mode
```

## 4.6 Saving and loading models (`state_dict`)

In PyTorch, we usually save **weights** (not the full Python object).

```python
import torch

torch.save(model.state_dict(), "mlp_weights.pt")

model2 = MLP()
model2.load_state_dict(torch.load("mlp_weights.pt", map_location="cpu"))
model2.eval()
```

---

# 5) The training loop

The standard training loop is always the same idea:

```text
1) Forward pass      y_hat = model(x)
2) Compute loss      loss  = loss_fn(y_hat, y)
3) Zero gradients    optimizer.zero_grad()
4) Backward pass     loss.backward()
5) Update weights    optimizer.step()
```

## 5.1 Example: learn a line (y = 2x)

```python
import torch
import torch.nn as nn

X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0], [10.0]])

model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(200):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 40 == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.6f}")

model.eval()
with torch.no_grad():
    print("x=6 ->", model(torch.tensor([[6.0]])).item())
```

## 5.2 Mini-batch training with `Dataset` + `DataLoader`

Real training is usually done in batches.

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

for epoch in range(3):
    for xb, yb in loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## 5.3 A clean training/eval pattern

This pattern is common in projects:

- `model.train()` for training
- `model.eval()` + `torch.no_grad()` for evaluation

---

# 6) Mini Lab: Binary classifier (with DataLoader)

Goal: build a binary classifier end-to-end.

*Figure: sigmoid maps logits (real numbers) to probabilities in (0,1). Source: Wikimedia Commons.*

## 6.1 Data

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import torch

X_np, y_np = make_classification(
    n_samples=2000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    n_classes=2,
    random_state=42,
)

X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_np, y_np, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train_np, dtype=torch.float32)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1)
```

## 6.2 Model (logits + `BCEWithLogitsLoss`)

Best practice for binary classification in PyTorch:

- Model outputs **logits** (raw scores)
- Use `nn.BCEWithLogitsLoss()` (more numerically stable than `Sigmoid + BCELoss`)

```python
import torch.nn as nn

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)  # logits
```

## 6.3 DataLoader

```python
from torch.utils.data import TensorDataset, DataLoader

train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256)
```

## 6.4 Training + evaluation

```python
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

model = BinaryClassifier(input_dim=10).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def accuracy_from_logits(logits, y_true):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    return (preds == y_true).float().mean()

for epoch in range(10):
    # ---- train ----
    model.train()
    total_loss = 0.0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = loss_fn(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)

    train_loss = total_loss / len(train_loader.dataset)

    # ---- eval ----
    model.eval()
    total_acc = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            total_acc += accuracy_from_logits(logits, yb).item() * xb.size(0)

    test_acc = total_acc / len(test_loader.dataset)
    print(f"Epoch {epoch+1:02d} | Train loss: {train_loss:.4f} | Test acc: {test_acc*100:.2f}%")
```

### Bonus exercises

1. Increase depth: add another hidden layer
2. Try `batch_size` = 16 vs 128 and compare training stability
3. Add L2 regularization: `torch.optim.Adam(..., weight_decay=1e-4)`
4. Add a validation split and track both validation loss and accuracy

---

# 7) One-page cheat sheet

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Tensors
x = torch.randn(4, 3, device=device)
b = torch.randn(3, device=device)
y = x + b                       # broadcasting

# Autograd
w = torch.tensor(2.0, device=device, requires_grad=True)
loss = (w * 3 - 6) ** 2
loss.backward()
print(w.grad)

# Model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        return self.net(x)

model = Net().to(device)

# Train loop skeleton
X = torch.randn(512, 10)
Y = torch.randn(512, 1)

ds = TensorDataset(X, Y)
loader = DataLoader(ds, batch_size=64, shuffle=True)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

model.train()
for xb, yb in loader:
    xb, yb = xb.to(device), yb.to(device)
    pred = model(xb)
    loss = loss_fn(pred, yb)
    opt.zero_grad()
    loss.backward()
    opt.step()
```

---

## Resources

- PyTorch 60-Minute Blitz: https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
- `torch.nn` docs: https://pytorch.org/docs/stable/nn.html
- Tensor basics tutorial: https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html

