

# Principal Component Analysis (PCA)

Principal Component Analysis (PCA) is one of the most important algorithms in **unsupervised learning** and **dimensionality reduction**.

It helps us simplify complex datasets by identifying the **most informative directions of variation in the data**.

Instead of analyzing hundreds of variables, PCA allows us to represent the same information using **a much smaller number of variables**.

---

# 1. What is PCA?

**Principal Component Analysis (PCA)** is a statistical method used to transform a dataset with many correlated variables into a smaller set of **uncorrelated variables called Principal Components**.

These components are ordered such that:

* **PC1 captures the maximum variance**
* **PC2 captures the next maximum variance**
* Each component is **orthogonal (90°) to the others**

In simple terms:

> PCA finds the directions where the data spreads the most and uses those directions as new axes.

---

# Visual Intuition of PCA

![Image](https://www.researchgate.net/profile/Christos-Dadousis/publication/267874264/figure/fig3/AS%3A282625416679455%401444394558503/Scatterplot-of-the-first-two-principal-components-PC1-vs-PC2-Principal-component.png)

![Image](https://miro.medium.com/1%2ANe5pL_8mMrdJiPvFpYeSUA.png)

![Image](https://miro.medium.com/0%2AjEIuEzHO-kIop-6-)

![Image](https://bryanhanson.github.io/LearnPCA/articles/Vig_04_Scores_Loadings_files/figure-html/rotate20-1.png)

Imagine a cloud of data points.

Instead of using the **original axes**, PCA rotates the axes to align with the **direction where data varies most**.

PC1 = direction of **maximum spread**
PC2 = second largest spread (perpendicular to PC1)

---

# 2. The Core Idea of PCA

The main principle behind PCA is:

> **Maximum Variance = Maximum Information**

If a feature has very little variance, it usually contains **very little useful information**.

Example:

| Feature   | Variance | Importance |
| --------- | -------- | ---------- |
| Age       | High     | Important  |
| Income    | High     | Important  |
| ID number | Zero     | Useless    |

PCA identifies the directions where **variance is highest** and compresses the dataset along those directions.

---

# 3. The "Shadow Projection" Intuition

Imagine a **3D object**.

You want to represent it in **2D**.

Depending on where you shine the light, the shadow may lose information.

![Image](https://www.researchgate.net/profile/Hamoud-Younes/publication/345602552/figure/fig1/AS%3A1028064637104131%401622121118891/PCA-Example-3D-to-2D.png)

![Image](https://www.researchgate.net/publication/221519024/figure/fig4/AS%3A339909983391749%401458052263311/Dimensionality-reduction-from-3D-to-2D.png)

![Image](https://www.scaler.com/topics/images/concept-of-pca.webp)

![Image](https://miro.medium.com/1%2ANe5pL_8mMrdJiPvFpYeSUA.png)

PCA mathematically finds the **best projection angle** so the shadow preserves **maximum structure** of the original object.

---

# 4. Why Do We Need PCA?

Modern datasets can contain hundreds or thousands of features.

This leads to several problems.

---

## 4.1 Curse of Dimensionality

As dimensions increase:

* Data becomes sparse
* Distance metrics become unreliable
* Models struggle to learn patterns

For example:

| Dimensions | Data Density     |
| ---------- | ---------------- |
| 2D         | Dense            |
| 10D        | Sparse           |
| 100D       | Extremely sparse |

This makes algorithms like **KNN almost useless** in very high dimensions.

---

## 4.2 Computational Cost

More features means:

* slower training
* larger memory usage
* slower predictions

Reducing dimensions speeds up models significantly.

---

## 4.3 Overfitting

When features >> samples

Models memorize noise instead of patterns.

PCA reduces noise by keeping only **important directions of variance**.

---

## 4.4 Visualization

Humans can visualize only:

* 2D
* 3D

PCA allows visualization of **100+ dimensional datasets**.

---

# 5. Step-by-Step PCA Algorithm

Let’s understand PCA mathematically.

---

# Step 1: Standardize the Data

PCA is sensitive to feature scale.

Example:

| Feature | Range       |
| ------- | ----------- |
| Salary  | 1000–100000 |
| GPA     | 0–4         |

Salary dominates variance.

So we standardize:

$$
z = \frac{x-\mu}{\sigma}
$$

After scaling:

* Mean = 0
* Standard deviation = 1

---

# Step 2: Compute Covariance Matrix

Covariance measures how two variables vary together.

$$
Cov(X,Y)=\frac{1}{n-1}\sum (x_i-\bar{x})(y_i-\bar{y})
$$

Where:
* $x_i$ and $y_i$ are individual daily data points for features $X$ and $Y$
* $\bar{x}$ (x-bar) is the **mean** (average) of all $X$ values
* $\bar{y}$ (y-bar) is the **mean** (average) of all $Y$ values
* $n$ is the total number of sample points

Interpretation:

| Covariance | Meaning                             |
| ---------- | ----------------------------------- |
| Positive   | Variables increase together         |
| Negative   | One increases while other decreases |
| Zero       | Independent                         |

Example:

### The Structure of a Covariance Matrix

For a dataset with $n$ features, the covariance matrix is an $n \times n$ symmetric matrix. The diagonal elements represent the **variance** of individual features, while the off-diagonal elements represent the **covariance** between pairs of features.

For a 2D dataset with features $X$ and $Y$, the matrix looks like this:

$$
\Sigma = \begin{bmatrix}
Var(X) & Cov(X, Y) \\
Cov(Y, X) & Var(Y)
\end{bmatrix}
$$

*(Note: $Cov(X, Y)$ always equals $Cov(Y, X)$, making the matrix symmetric).*

### Numerical Example

Imagine a small dataset measuring the **Study Hours** ($X$) and **Test Scores** ($Y$) of students:

| Student | Study Hours (X) | Test Score (Y) |
| ------- | --------------- | -------------- |
| A       | 2               | 50             |
| B       | 4               | 70             |
| C       | 6               | 90             |

Let's calculate the values step-by-step applying **Step 1 (Standardization)** and **Step 2 (Covariance)**.

**1. Standardize the features (Calculate Z-scores)**
First, we calculate the **mean** ($\bar{x}$) and **standard deviation** ($\sigma$) for both features. 
*(Note: we use the formula for sample standard deviation: $\sigma = \sqrt{\frac{\sum (x_i - \bar{x})^2}{n-1}}$)*

**For Study Hours (X):**
* Mean ($\bar{x}$): $(2 + 4 + 6) / 3 = 4$
* Variance: $\frac{(2 - 4)^2 + (4 - 4)^2 + (6 - 4)^2}{3 - 1} = \frac{4 + 0 + 4}{2} = 4$
* Standard Deviation ($\sigma_x$): $\sqrt{4} = 2$

**For Test Scores (Y):**
* Mean ($\bar{y}$): $(50 + 70 + 90) / 3 = 70$
* Variance: $\frac{(50 - 70)^2 + (70 - 70)^2 + (90 - 70)^2}{3 - 1} = \frac{400 + 0 + 400}{2} = 400$
* Standard Deviation ($\sigma_y$): $\sqrt{400} = 20$

Formula for standardization: $z = \frac{\text{value} - \text{mean}}{\sigma}$

| Student | Standardized X ($Z_x$) | Standardized Y ($Z_y$) |
| ------- | ---------------------- | ---------------------- |
| A       | (2 - 4) / 2 = -1       | (50 - 70) / 20 = -1    |
| B       | (4 - 4) / 2 = 0        | (70 - 70) / 20 = 0     |
| C       | (6 - 4) / 2 = 1        | (90 - 70) / 20 = 1     |

**Notice:** The means of $Z_x$ and $Z_y$ are now exactly $0$.

**2. Calculate Variance of Standardized X ($Var(Z_x)$)**
Formula: $\frac{1}{n-1} \sum (z_{xi} - 0)^2$
* Student A: $(-1)^2 = 1$
* Student B: $(0)^2 = 0$
* Student C: $(1)^2 = 1$
* Sum = $2$
* $Var(Z_x) = 2 / (3 - 1) = 1.0$

**3. Calculate Variance of Standardized Y ($Var(Z_y)$)**
Formula: $\frac{1}{n-1} \sum (z_{yi} - 0)^2$
* Student A: $(-1)^2 = 1$
* Student B: $(0)^2 = 0$
* Student C: $(1)^2 = 1$
* Sum = $2$
* $Var(Z_y) = 2 / (3 - 1) = 1.0$

**4. Calculate Covariance between Standardized X and Y ($Cov(Z_x,Z_y)$)**
Formula: $\frac{1}{n-1} \sum (z_{xi} - 0)(z_{yi} - 0)$
* Student A: $(-1) \times (-1) = 1$
* Student B: $(0) \times (0) = 0$
* Student C: $(1) \times (1) = 1$
* Sum = $2$
* $Cov(Z_x,Z_y) = 2 / (3 - 1) = 1.0$

The resulting Covariance Matrix (which is also the Correlation Matrix!) is:

$$
\Sigma = \begin{bmatrix}
1.0 & 1.0 \\
1.0 & 1.0
\end{bmatrix}
$$

**How to read this matrix:**
* **Diagonals (1.0):** The variance of standardized features is always exactly 1.
* **Off-Diagonals (1.0):** The covariance is perfectly 1.0 (indicating a 100% positive linear correlation). As standard study hours increase, standard test scores increase identically!

---

### Covariance vs. Correlation (Important Distinction)

Students often confuse Covariance with **Correlation**. While both measure the relationship between two variables, they are fundamentally different in scale.

| Feature | Covariance | Correlation (Pearson) |
| :--- | :--- | :--- |
| **Formula Measure** | Raw unscaled relationship | Normalized relationship |
| **Range of Values** | $-\infty$ to $+\infty$ | $-1$ to $+1$ |
| **Scale Dependence** | Highly dependent on units (Dollars vs Cents changes the value) | **Scale-free** (Unitless) |
| **Interpretability** | Tells direction (+ or -) but not strength clearly | Tells both direction and strength precisely |

**Why this matters for PCA:**
If we computed PCA on **unscaled** raw data, features with massive absolute numbers (like Test Scores) would completely dominate the covariance matrix, hiding the true variance of smaller features (like Study Hours). 

By performing **Step 1 (Standardization)** first, our new Covariance Matrix becomes exactly equal to a **Correlation Matrix**. This ensures PCA treats all features fairly, regardless of their original units!

---

# Step 3: Eigenvectors and Eigenvalues

This is the mathematical heart of PCA. 

We decompose the covariance matrix into **Eigenvectors** and **Eigenvalues**.

### What are they? (Intuition)
Imagine stretching a piece of rubber. Most points on the rubber move in unpredictable curves. However, there are specific fixed directions where the rubber *only* stretches in a straight line, without rotating. 

* **Eigenvector ($v$):** The *direction* of that straight-line stretch (the axis of maximum variance).
* **Eigenvalue ($\lambda$):** The *amount* of stretch in that direction (the magnitude of variance).

In PCA, the Eigenvectors become our new "Principal Components", and the Eigenvalues tell us how important each component is!

---

### How to Calculate Them (Simplified Step-by-Step)

If we start with our Standardized Covariance Matrix:
$$
\Sigma = \begin{bmatrix} 1.0 & 1.0 \\ 1.0 & 1.0 \end{bmatrix}
$$

We want to find two things:
1. **Eigenvalues ($\lambda$)**: The "amount of variance".
2. **Eigenvectors ($v$)**: The "direction".

---

**Step 1: Finding the Eigenvalues (The Amount of Variance)**
To find $\lambda$, we solve the **characteristic equation**: $det(\Sigma - \lambda I) = 0$

$$
det \left( \begin{bmatrix} 1.0 & 1.0 \\ 1.0 & 1.0 \end{bmatrix} - \lambda \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \right) = 0
$$

$$
det \begin{bmatrix} 1 - \lambda & 1 \\ 1 & 1 - \lambda \end{bmatrix} = 0
$$

**Cross-multiplying:** $(1 - \lambda)(1 - \lambda) - (1)(1) = 0$
$1 - 2\lambda + \lambda^2 - 1 = 0$
$\lambda^2 - 2\lambda = 0$
$\lambda(\lambda - 2) = 0$

This matrix gives us two answers (our Eigenvalues!):
* $\lambda_1 = 2$ 
* $\lambda_2 = 0$ 

*Rule: The highest eigenvalue becomes Principal Component 1 (PC1) because it captures the most information!*

---

**Step 2: Finding the Eigenvectors (The Direction)**
Now that we know our biggest variance is $2$, we ask the matrix: 
*"Which direction gives us this stretch of 2?"*

We plug $\lambda_1 = 2$ back into the equation: $(\Sigma - \lambda I)v = 0$

$$
\begin{bmatrix} 1 - 2 & 1 \\ 1 & 1 - 2 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$

$$
\begin{bmatrix} -1 & 1 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$

This gives us the system of equations:
$-v_1 + v_2 = 0 \Rightarrow v_1 = v_2$
$v_1 - v_2 = 0 \Rightarrow v_1 = v_2$

This simply means that for every **1 step** we move along the X-axis (Study Hours), we must move **1 step** along the Y-axis (Test Scores).

This ratio gives us our Eigenvector!
$$
v = \begin{bmatrix} 1 \\ 1 \end{bmatrix}
$$



### What did we just learn?
We mathematically proved that **Principal Component 1 (PC1)** points exactly diagonally at $\begin{bmatrix} 1 \\ 1 \end{bmatrix}$. Because we standardized the data first, PCA treats both features equally instead of being biased towards the large numbers in Test Scores!

Because PC1's eigenvalue is $2$ (compared to $0$ for the other direction), we know that **nearly ALL the useful information in this dataset lies along this single axis!** We can drop the other axis entirely and not lose any data context.

---

# Step 4: Sort Principal Components

We rank eigenvectors by eigenvalues.

Highest eigenvalue → PC1
Second highest → PC2

Example:

| Component | Variance |
| --------- | -------- |
| PC1       | 60%      |
| PC2       | 25%      |
| PC3       | 10%      |
| PC4       | 5%       |

If we keep PC1 + PC2:

We retain **85% of the information**.

---

# Step 5: Project Data to New Space

Finally we transform the data.

$$
Z = XW
$$

Where

X = original data
W = eigenvector matrix

Result:

New dataset with fewer dimensions.

---

# PCA Transformation Visualization

![Image](https://miro.medium.com/0%2AjEIuEzHO-kIop-6-)

![Image](https://miro.medium.com/1%2A_wcd4AGrcovM0m_WypIYtQ.png)

![Image](https://miro.medium.com/v2/resize%3Afit%3A1400/0%2Al8A5yMWy5DYzox1F)

![Image](https://media.licdn.com/dms/image/v2/D4D12AQF61SUXClGqIg/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1660108512262?e=2147483647\&t=NNFCMNh_gHqd7agZERlwm11pudhYNdkt7t0vRvQxOSQ\&v=beta)

Instead of dropping features randomly, PCA **rotates the coordinate system**.

---

# 6. Explained Variance Ratio

Explained variance tells us how much information each component captures.

$$
ExplainedVariance = \frac{\lambda_i}{\sum \lambda}
$$

Example:

| Component | Variance Explained |
| --------- | ------------------ |
| PC1       | 45%                |
| PC2       | 30%                |
| PC3       | 15%                |
| PC4       | 10%                |

PC1 + PC2 retain **75% of dataset information**.

---

# Scree Plot Visualization

![Image](https://www.researchgate.net/publication/338833359/figure/fig1/AS%3A851904309522434%401580121223698/A-scree-plot-for-explained-variance-and-eigenvalues-for-the-ten-Principal-Components.png)

![Image](https://upload.wikimedia.org/wikipedia/commons/a/ac/Screeplotr.png)

![Image](https://miro.medium.com/v2/resize%3Afit%3A1210/1%2ANx8nLPdHmAtgWopOLa76Bg.png)

![Image](https://hyperspy.org/hyperspy-doc/current/_images/screeplot_elbow_method.png)

A **Scree Plot** helps determine how many components to keep.

We usually stop when the curve starts flattening.

---

# 7. Interpretation of Principal Components

Principal Components are **linear combinations** of original features.

Example:

Original features

Age
Income
Spending score

PC1 might be:

$$
PC1 = 0.5 \text{ Age} + 0.7 \text{ Income} + 0.2 \text{ SpendingScore}
$$

This makes interpretation difficult.

This is called **loss of interpretability**.

---

# 8. PCA Geometric Meaning

Geometrically PCA does two things:

1. **Rotates the coordinate system**
2. **Projects data onto lower dimensions**

This projection preserves **maximum variance**.

---

# 9. Advantages of PCA

Reduces dimensionality

Removes redundant features

Speeds up ML models

Helps visualization

Reduces noise

---

# 10. Limitations of PCA

Loss of interpretability

Assumes linear relationships

Sensitive to scaling

May discard important low-variance features

---

# 11. Real World Applications

---

### Image Compression

Images contain millions of pixels.

Many pixels are highly correlated.

PCA compresses image while preserving structure.

Example:

Eigenfaces algorithm.

---

### Finance

Used in **portfolio risk analysis**.

Reduces many correlated financial indicators into few components.

---

### Genomics

DNA datasets contain thousands of genes.

PCA helps identify gene clusters related to diseases.

---

### Data Visualization

Common workflow:

High dimensional dataset → PCA → 2D scatter plot → cluster analysis.

---

# 12. PCA vs Other Methods

| Method  | Purpose                             |
| ------- | ----------------------------------- |
| PCA     | Dimensionality reduction            |
| K-Means | Clustering                          |
| t-SNE   | Visualization                       |
| LDA     | Supervised dimensionality reduction |

---

# Interactive Questions

### Question 1

Why must PCA features be standardized?

Answer:

Features with large scale dominate variance.

---

### Question 2

Why must principal components be orthogonal?

Answer:

To ensure no redundancy between components.

---

### Question 3

Does PCA use target labels?

Answer:

No.

PCA is **unsupervised**.

---


