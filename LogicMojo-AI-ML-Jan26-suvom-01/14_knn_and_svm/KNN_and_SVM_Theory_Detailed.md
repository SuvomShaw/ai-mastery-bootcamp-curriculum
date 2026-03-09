# K-Nearest Neighbors (KNN) & Support Vector Machines (SVM)


| Algorithm | Learning Style            | Core Idea                                   |
| --------- | ------------------------- | ------------------------------------------- |
| **KNN**   | Instance-based learning   | Similar data points lie close together      |
| **SVM**   | Margin-based optimization | Find the boundary that maximizes separation |

Understanding both gives students insight into **two major ML philosophies**:

1. **Memory-based learning**
2. **Optimization-based learning**

---

# PART 1 — K NEAREST NEIGHBORS (KNN)

---

## 1. What is KNN?

**K-Nearest Neighbors (KNN)** is a supervised learning algorithm that predicts the class or value of a new data point based on the **closest training data points in feature space**.

The algorithm follows a simple idea:

> **Points that are close to each other are likely to belong to the same category.**

This idea comes from **similarity learning**.

Example intuition:

If your neighbors are mostly doctors and engineers, chances are your social circle belongs to those professions.

---

## 2. Visual Intuition of KNN

![Image](https://miro.medium.com/v2/resize%3Afit%3A1182/0%2AsYMSaIon56Qng2hF.png)

![Image](https://i.sstatic.net/jz0hd.jpg)

![Image](https://images.openai.com/static-rsc-3/5n-_kmBdJk15FjnwaXZSofyBRwVI9vIzMuTRY1Lu5nd-Crc1GuOv_mwqMFEhU8SiCg2Dn2X-Tr12-a9R3DyE6TEA-ZYaERhw7z9S1wLA12w?purpose=fullsize\&v=1)

![Image](https://images.openai.com/static-rsc-3/Ar9X_cVSxFyd3dWni2fzogTKEx6i0cOxEZGvW4DQvqMao-ziiSXy4WXpDIR9ATAF3j_fVRfcKoDdb618hcdBLddmhiq8AkzCsCRA-mF_k_o?purpose=fullsize\&v=1)

Imagine a 2D feature space:

Blue = Class A
Red = Class B

A new point arrives.

KNN simply asks:

> Who are the **K closest neighbors**?

If most neighbors are **blue → classify blue**.

---

# 3. Why KNN Works (Theory)

KNN is based on the assumption of **local smoothness**.

This means:

> Data points that are close together tend to have similar outputs.

Mathematically:

If two points $(x_i)$ and $(x_j)$ are close

then $f(x_i) \approx f(x_j)$

This assumption holds true in many real-world datasets:

Examples:

| Problem             | Why it works                        |
| ------------------- | ----------------------------------- |
| Recommender systems | Similar users like similar products |
| Medical diagnosis   | Similar symptoms → similar disease  |
| Image recognition   | Similar pixels → similar object     |

---

# 4. Distance Metrics (How KNN Measures Similarity)

To determine neighbors, KNN computes **distance between points**.

---

## Euclidean Distance

Most common distance metric.

$$
d(x,y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}
$$

Geometric interpretation:

Straight line between two points.

Example:

Height & Weight features.

---

## Manhattan Distance

$$
d(x,y) = \sum |x_i - y_i|
$$

This measures movement along grid paths.

Used in:

* high dimensional spaces
* sparse data

---

## Minkowski Distance (General Form)

$$
d(x,y) = \left(\sum |x_i-y_i|^p \right)^{1/p}
$$

If:

| p   | Metric    |
| --- | --------- |
| p=1 | Manhattan |
| p=2 | Euclidean |

---

# 5. Full KNN Algorithm (Step by Step)

Training Phase:

There is **no real training**.

The model simply **stores the dataset**.

Prediction Phase:

Step 1
Choose **K**

Step 2
Compute distance from test point to every training point

Step 3
Sort by distance

Step 4
Select **K nearest neighbors**

Step 5
Aggregate results

Classification:

Majority vote

Regression:

Average value

---

# 6. Example (Classification)

Suppose we classify **loan approval**.

Features:

| Income | Credit Score |
| ------ | ------------ |

Training data:

| Income | Credit | Class   |
| ------ | ------ | ------- |
| 40k    | 600    | Reject  |
| 60k    | 700    | Approve |
| 65k    | 720    | Approve |
| 50k    | 650    | Reject  |

New applicant:

Income = 62k
Credit = 710

Compute distances.

Closest neighbors mostly **Approve → prediction Approve**

---

# 7. Decision Boundary in KNN

KNN produces **highly non-linear boundaries**.

Because classification depends on **local neighbors**.

![Image](https://i.sstatic.net/jz0hd.jpg)

![Image](https://i.sstatic.net/e9ved.png)

![Image](https://miro.medium.com/1%2Anv2sxUuhFDzYqKnWEHJ6WA.png)

![Image](https://i.sstatic.net/LZWS8.png)

Observation:

Small K → complex boundary
Large K → smoother boundary

---

# 8. Choosing K (Bias vs Variance)

| K value | Effect       |
| ------- | ------------ |
| Small K | Overfitting  |
| Large K | Underfitting |

Example:

K = 1

Model memorizes data.

Very sensitive to noise.

K = large

Model predicts majority class.

Rule of thumb:

$K \approx \sqrt{N}$

---

# 9. Major Problem of KNN (Curse of Dimensionality)

When dimensions increase:

Distances become meaningless.

Example:

In 2D space
Points cluster nicely.

In 100D space

Everything becomes **equally distant**.

![Image](https://media.licdn.com/dms/image/v2/D4D12AQF0V9o53Sek3w/article-cover_image-shrink_720_1280/B4DZhesQ9VGsAI-/0/1753935342419?e=2147483647\&t=L6Ih7TRjOVIzvj3J_vngmRFrLcmPe_6bZlYV8b713Jg\&v=beta)

![Image](https://miro.medium.com/0%2Acz8qnERPVMqqT8Fe.jpeg)

![Image](https://i.sstatic.net/f1WOm.png)

![Image](https://i.sstatic.net/PfMe3.png)

This is why KNN struggles in **high dimensional datasets** like text.

---

# 10. Advantages of KNN

Very simple

No training time

Non-parametric

Works well for small datasets

---

# 11. Disadvantages of KNN

Prediction is slow

Memory intensive

Sensitive to noise

Poor in high dimensions

Requires feature scaling

---

# 12. Real World Applications

### Recommendation Systems

Netflix style recommendations.

### Medical Diagnosis

Patients with similar symptoms.

### Image Recognition

Digit recognition (MNIST).

### Fraud Detection

Transactions similar to fraudulent ones.

---

# PART 2 — SUPPORT VECTOR MACHINES (SVM)

---

# 1. What is SVM?

Support Vector Machine is a supervised learning algorithm that finds the **optimal separating boundary between classes**.

The key concept:

> Choose the boundary that **maximizes margin**.

Margin = distance between decision boundary and closest data points.

---

# 2. Visual Intuition of SVM

![Image](https://miro.medium.com/1%2AbxxbmAIbn8iqv7zJlk576Q.png)

![Image](https://miro.medium.com/v2/resize%3Afit%3A1400/1%2AqGl9Os6nA82aQrDUhZ7zOQ.png)

![Image](https://www.researchgate.net/publication/335409647/figure/fig3/AS%3A948681440235524%401603194689539/sualization-of-the-hyperplane-generated-by-a-SVM-model-separating-the-samples-of-two.ppm)

![Image](https://www.researchgate.net/publication/327743769/figure/fig1/AS%3A672445979705344%401537335023562/llustration-of-hyperplane-in-linear-SVM-M-2-w-is-the-large-margin-as-show-in-Fig.png)

Blue = Class A
Red = Class B

Many separating lines exist.

But SVM selects the one with **largest margin**.

---

# 3. Why Maximum Margin?

Large margin means:

Better generalization.

Small margin boundaries are sensitive to noise.

Mathematically:

Margin = $\frac{2}{||w||}$

Maximizing margin = Minimizing $||w||$

---

# 4. Mathematical Formulation

Hyperplane equation: $w^T x + b = 0$

Where

w = weight vector
b = bias

Constraints: $y_i (w^T x_i + b) \ge 1$

Optimization objective:

$$ \min \frac{1}{2} ||w||^2 $$

subject to constraints.

This is a **convex optimization problem**.

---

# 5. Support Vectors

Support vectors are the **critical points closest to the boundary**.

Only these points determine the model.

Removing other points **does not affect boundary**.

---

# 6. Non-Linear Data Problem

Real datasets are rarely linearly separable.

Example:

Circle inside circle.

Linear boundary fails.

Solution:

**Kernel Trick**

---

# 7. Kernel Trick

Idea:

Map data to **higher dimensional space**.

Where linear separation becomes possible.

![Image](https://miro.medium.com/1%2A_Uhpj662QpxoIa8qlPYJ9A.png)

![Image](https://www.researchgate.net/publication/225415339/figure/fig1/AS%3A648614107955200%401531653062199/Decision-boundary-by-SVM-with-RBF-kernel-function.png)

![Image](https://miro.medium.com/1%2AzWzeMGyCc7KvGD9X8lwlnQ.png)

![Image](https://miro.medium.com/1%2ADmQEIIz29UulOmmxDx5DrQ.jpeg)

Example:

2D → 3D transformation.

After transformation

Linear plane separates data.

---

# 8. Common Kernels

### Linear Kernel

$$
K(x_i,x_j) = x_i^T x_j
$$

Used when data is linear.

---

### Polynomial Kernel

$$
K(x_i,x_j) = (x_i^T x_j + c)^d
$$

Captures polynomial relationships.

---

### RBF Kernel (Most popular)

$$
K(x_i,x_j) = e^{-\gamma ||x_i-x_j||^2}
$$

Works extremely well for complex patterns.

---

# 9. Soft Margin SVM

![Hard vs Soft Margin SVM](https://miro.medium.com/max/1465/1*PiGj6vEyBhxbXfK4bzwwTg.png)

![Soft Margin with Slack Variables](https://media.geeksforgeeks.org/wp-content/uploads/20240219195819/svm2.webp)

Real data contains noise.

SVM allows misclassification using slack variables.

Objective becomes:

$$ \min \frac{1}{2}||w||^2 + C\sum \xi_i $$

Where

C controls tradeoff.

Small C

→ wider margin
→ more errors allowed

Large C

→ narrow margin
→ fewer errors allowed

---

# 10. SVM vs KNN (Important Comparison)

| Feature                  | KNN    | SVM       |
| ------------------------ | ------ | --------- |
| Training time            | None   | High      |
| Prediction time          | Slow   | Fast      |
| Works in high dimensions | Poor   | Excellent |
| Memory usage             | High   | Low       |
| Interpretability         | Simple | Moderate  |

---

# 11. Applications of SVM

### Text Classification

Spam filtering.

### Image Recognition

Face detection.

### Bioinformatics

Gene classification.

### Financial Forecasting

Credit scoring.

---

# 12. Questions

### Question 1

Why must features be scaled for KNN?

Answer:

Distance depends on scale.

Example:

Income = 100000
Age = 25

Income dominates distance.

---

### Question 2

Why does SVM use only support vectors?

Answer:

Optimization solution depends only on boundary points.

Interior points have zero influence.

---

### Question 3

Why does RBF kernel work so well?

Answer:

It can create highly flexible boundaries that adapt locally to data structure.

---
