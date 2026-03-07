# Logistic Regression – Student Notes

**AI & ML Bootcamp | Module: Classification**

---

## Contents (Quick)

- 1. What is Logistic Regression?
- 2. Why Not Linear Regression?
- 3. The Geometric Derivation (Linear Algebra)
- 4. Understanding Log-Loss (The Cost Function)
- 5. The Probabilistic Connection (Sigmoid, Odds, and Log-Odds)
- 6. Decision Boundary
- 7. Regularization (L1 & L2)
- 8. Confusion Matrix
- 9. Evaluation Metrics
- 10. Class Imbalance
- 11. Multiclass Logistic Regression
- 12. When to Use / Not Use Logistic Regression
- 13. Key Takeaways

---

## Notation (Symbols used)

*   \(X\): feature matrix (inputs)
*   \(x\): one sample's feature vector
*   \(y\): actual class label (+1 or -1 for derivation; 1 or 0 for probability)
*   \(\hat{p}\): predicted probability of class 1
*   \(\hat{y}\): predicted class label
*   \(z\): linear score/distance (\(\theta^T x\))
*   \(\sigma(z)\): sigmoid function
*   \(\theta\): model weights (normal vector to the hyperplane, including intercept)
*   \(m\): number of training examples
*   \(\alpha\): learning rate (gradient descent step size)
*   \(\lambda\): regularization strength

---

## 1. What is Logistic Regression?

### Concept
Despite its name, Logistic Regression is a **classification** algorithm, not a regression one. It is used to separate data points into distinct categorical classes (e.g., "Positive" vs "Negative").

### Definition
**Logistic Regression** is an algorithm that searches and finds the single "best" separating line (or hyperplane in higher dimensions) to cleanly divide two classes. 

### Notes / Intuition
Instead of predicting a raw continuous number (like "Price = \$200k"), it measures the *distance* a data point falls from the dividing boundary.

*   **Input**: Features (Email words, Sender history)
*   **Geometric Output**: Distance from the boundary.
*   **Decision**: If distance is positive $\rightarrow$ Spam; if distance is negative $\rightarrow$ Not Spam.

### Examples
*   **Spam Detection**: Spam (+1) vs Not Spam (-1)
*   **Medical Diagnosis**: Diseased (+1) vs Healthy (-1)
*   **Banking**: Default (+1) vs Paid (-1)

---

## 2. Why Not Linear Regression?

### Concept
Linear Regression fits a straight continuous line to predict quantities. For explicit classification, a straight regression line fails because it is unbounded and profoundly sensitive to extreme outliers.

### Visual explanation
Imagine a 1D plot of Tumor Size vs Malignancy Status (-1 or +1).
*   **Linear Regression**: Tries to draw a slanted line connecting the -1s and +1s. A single massive tumor outlier far to the right will completely drag the line down, causing it to misclassify perfectly normal smaller tumors in the middle.
*   **Logistic Regression**: Specifically optimizes a vertical boundary split (a separator) that ignores how far away an outlier is, so long as it remains on the "correct" side of the fence.

---

## 3. The Geometric Derivation (Linear Algebra)

This section builds the core Logistic Regression optimization formula strictly using vectors, planes, and distances. **There is no probability required to build the model!**

### 3.1 Step 1: The Hyperplane Equation
We want to draw a flat boundary (a line in 2D, a plane in 3D, a hyperplane in ND) that splits our data. From linear algebra, any plane passing through an origin can be defined by its **normal vector** $\theta$.

The distance from any data point vector $x$ to this plane is proportional to the **dot product**:
$$ z = \theta^T x $$

*   If $z > 0$, the point is pointing in the *same direction* as the normal vector (Side A).
*   If $z < 0$, the point is pointing in the *opposite direction* (Side B).
*   If $z = 0$, the point is sitting *exactly on the boundary*.

### 3.2 Step 2: Formulating the "Margin"
Let's label our two operational classes mathematically as:
$$ y \in \{-1, +1\} $$

Since $y$ dictates the "true" side of the plane, and $z$ ($\theta^T x$) dictates the "predicted" side of the plane, we can multiply them to create incredibly useful geometry called the **Functional Margin ($m$)**:
$$ m = y (\theta^T x) $$

![Deriving the Geometric Boundary: Hyperplane and Margins](https://upload.wikimedia.org/wikipedia/commons/b/b5/Svm_separating_hyperplanes_%28SVG%29.svg)

What does this single multiplication accomplish algebraically?
1.  **Correct & Far (Good)**: True $y=1$, Predicted $z=5 \implies m = +5$ (Large Positive)
2.  **Correct & Far (Good)**: True $y=-1$, Predicted $z=-5 \implies m = +5$ (Large Positive)
3.  **Wrong (Bad)**: True $y=1$, Predicted $z=-2 \implies m = -2$ (Negative)

**The Universal Rule**: If the margin $m$ is positive, the point is classified perfectly correctly. If $m$ is negative, it is an error. We want to maximize $m$ for every single point.

### 3.3 Step 3: The Ideal (But Impossible) Cost Function
Ideally, we want a cost function that gives a penalty of `0` if $m > 0$ (correct) and a penalty of `1` if $m < 0$ (incorrect). This is the **0-1 Step Loss**.

**The Mathematical Problem**: The 0-1 step loss is shaped like a staircase. It is entirely flat everywhere, making its derivative (gradient) exactly zero. If the gradient is zero, Gradient Descent cannot move! We mathematically cannot optimize it.

### 3.4 Step 4: Creating a Mathematical "Proxy" Curve
Since we cannot use the stiff staircase function, we need to invent a smooth, continuous mathematical curve that tightly "hugs" the staircase. This curve must have two geometrical rules:
1.  As $m \rightarrow +\infty$ (very correct), the penalty must smoothly approach 0.
2.  As $m \rightarrow -\infty$ (very wrong), the penalty must scale linearly upwards.

The smoothest algebraic function that perfectly satisfies this behavior is the **Logistic Loss Curve**:
$$ \text{Loss}(m) = \log(1 + e^{-m}) $$

![Approximating Step Loss with Logistic Loss](https://scikit-learn.org/stable/_images/sphx_glr_plot_sgd_loss_functions_001.png)

Let's plug our margin back in!
$$ \text{Cost for one point} = \log(1 + e^{-y (\theta^T x)}) $$

*(Graphing this equation reveals a beautifully smooth slope that gently penalizes small mistakes, heavily penalizes massive mistakes, and grants zero reward to already-correct points. Best of all: it is perfectly differentiable everywhere!)*

### 3.5 Step 5: The Final Cost Function (Log-Loss)
To evaluate the whole model, we take the average of this geometric loss across all $m$ training points to create our ultimate Cost Function $J(\theta)$:

$$ J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \log(1 + e^{-y^{(i)} \theta^T x^{(i)}}) $$

---

## 4. Understanding Log-Loss (The Cost Function)

### The Optimization: Gradient Descent
To find the best plane (the best $\theta$), we must move down the slope of $J(\theta)$. We calculate the partial derivatives. By employing the chain rule on that $\log(1+e^{-z})$ function, it gracefully simplifies into a beautifully clean algebraic update step:

$$ \nabla J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \left( \frac{-y^{(i)} x^{(i)}}{1 + e^{y^{(i)} \theta^T x^{(i)}}} \right) $$

We iteratively update our normal vector to slightly tilt the separating plane closer to perfection:
$$ \theta = \theta - \alpha \nabla J(\theta) $$

### Why not just use Mean Squared Error (MSE)?
Mathematical purists ask: "Why did we invent that weird log curve instead of using MSE ($y - \theta^T x)^2$?"
*   If a point is $y=+1$, and our plane scores it accurately at $z = +100$, that point is *incredibly correct*! 
*   However, MSE would calculate the error as $(1 - 100)^2 = 9801$.
*   Mean Squared Error actually severely penalizes points for being "too correct" (too far from the plane). We want to encourage points to be as far onto the correct side of the boundary as possible!

---

## 5. The Probabilistic Connection (Sigmoid, Odds, and Log-Odds)

Now that we have derived the geometric model, an incredible mathematical coincidence occurs. It turns out that the geometric distance to the boundary perfectly maps to a strict statistical probability!

### 5.1 The Sigmoid Function (Translating Distance to Probability)
If a point is resting perfectly on the boundary ($\theta^T x = 0$), we are 50% confident about its class. The further positive it goes, the closer to 100% confident we get. We translate the unbounded distance $(-\infty, +\infty)$ into a strict probability $[0, 1]$ using the **Sigmoid** activation function:

$$ \hat{p} = \sigma(z) = \frac{1}{1 + e^{-z}} $$

*   **Squashing**: No matter how large the algebraic score $z$ is, $\sigma(z)$ stays trapped between 0 and 1.
*   **Center**: $\sigma(0) = 0.5$ (The boundary / uncertain point).

![Sigmoid Probability](https://developers.google.com/static/machine-learning/crash-course/logistic-regression/images/sigmoid_function_with_axes.png)

### 5.2 Odds and Log-Odds
Once we accept $\hat{p}$ as a bounded probability, we can explore its inverse relationship mathematically.

*   **Probability**: "This horse has an 80% chance of winning (0.8)."
*   **Odds**: "If it plays 5 games, it wins 4 and loses 1. The odds of winning are 4 to 1 (4:1)."
*   Probability is `Wins / Total Games`. Odds are `Wins / Losses`.

#### The Translation
1.  **Probability ($p$)** is stuck between **0 and 1**. 
2.  **Odds ($\frac{p}{1-p}$)** are stuck between **0 and $+\infty$**. 
3.  **Log-Odds ($\log(\frac{p}{1-p})$)** stretches infinitely from **$-\infty$ to $+\infty$**!

If we take the probability equation $p = \frac{1}{1 + e^{-z}}$ and rigorously solve backwards for $z$, we mathematically prove:
$$ z = \log\left(\frac{p}{1-p}\right) $$
$$ \theta^T x = \text{Log-Odds} $$

*This is the mathematical magic*: The raw linear algebraic score ($z$) output by the hyperplane is literally identical to the statistical "Log-Odds" of the event occurring!

### 5.3 Interpretation of Coefficients ($\theta$)
*   If $\theta_1 = 0.7$: Increasing feature $x_1$ by 1 unit increases the geometric distance $z$ by 0.7, meaning the **log-odds** increase by 0.7.
*   Since $\text{Odds} = e^{\text{Log-Odds}}$, an increase of 0.7 in log-odds multiplies the **odds** by $e^{0.7} \approx 2.0$.
*   *Plain English*: "A one-unit increase in this specific feature **doubles** the odds of the target outcome."

---

## 6. Decision Boundary

### Theory
The decision boundary is the multidimensional line where the model is functionally uncertain ($z = 0$, or $\hat{p} = 0.5$).

### Definition
The boundary occurs exactly when our dot-product calculation neutralizes:
$$ \theta_0 + \theta_1 x_1 + \theta_2 x_2 ... = 0 $$

*   **2D Data**: The boundary is a **Line** separating the two classes.
*   **3D Data**: The boundary is a **Plane**.
*   **Higher D**: The boundary is a **Hyperplane**.
*   Points on one side of the line are classified as 1, points on the other side as 0.

![Decision Boundary](https://scipython.com/media/old_blog/logistic_regression/decision-boundary.png)

---

## 7. Regularization (Prevent Overfitting)

Think of a model that mathematically memorizes the exact distance to every single training data point as a student purely memorizing past exam papers. Regularization is a **teacher adding an algebraic penalty** for overly literal, complex answers enforcing generalized behavior rules.

### L2 Regularization (Ridge) - "The Dimmer Switch"
$$ J = \text{LogLoss} + \frac{\lambda}{2m}\sum \theta_j^2 $$
*   **Analogy**: Reduces the brightness (weight impact) of all geometric dimensions without zeroing them blindly.
*   **Use when**: Most features provide value, and you need to prevent extreme vector tilting. (Default choice).

### L1 Regularization (Lasso) - "The On/Off Switch"
$$ J = \text{LogLoss} + \frac{\lambda}{m}\sum |\theta_j| $$
*   **Analogy**: An on/off switch. Evaluates weak dimensions and zeroes out that dimension entirely (flattening the plane across that axis).
*   **Use when**: You have 1000 features but suspect only 10 matter. 

*(Why L1 hits zero and L2 does not? L2 applies a weak parabolic squaring pressure near zero, while L1 applies a constant sharp absolute pressure forcing values all the way down identical to a geometric diamond constraint).*

---

## 8. Confusion Matrix

Visually charts performance layout. 

| | **Predicted Negative (0)** | **Predicted Positive (1)** |
| :--- | :--- | :--- |
| **Actual Negative (0)** | **TN** (True Negative) | **FP** (False Positive) <br> *(Type I Error)* |
| **Actual Positive (1)** | **FN** (False Negative) <br> *(Type II Error)* | **TP** (True Positive) |

*   **TP**: Predicted COVID, they have COVID.
*   **TN**: Predicted Healthy, they are Healthy.
*   **FP**: Predicted COVID, they are Healthy. (False Alarm!)
*   **FN**: Predicted Healthy, they have COVID. (Dangerous Miss!)

---

## 9. Evaluation Metrics (Crucial for Imbalanced Data)

### 1. Accuracy
*   $\frac{TP + TN}{\text{Total}}$
*   *Avoid heavily on imbalanced datasets* (e.g. 99% of transactions are normal. Model strictly guesses "Normal". It achieves 99% accuracy but spots zero fraud).

### 2. Precision (Quality of Positives)
*   $\frac{TP}{TP + FP}$
*   Use when **False Positives are expensive/annoying**. Focuses purely on eliminating False Alarms.

### 3. Recall / Sensitivity (Quantity of Positives)
*   $\frac{TP}{TP + FN}$
*   Use when **False Negatives are dangerous/fatal**. Prioritizes finding every true positive, regardless of the False Alarm rate.

### 4. F1 Score (The Balance)
*   $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
*   The best metric for imbalanced classes when you need a harmonized baseline between Precision and Recall constraints.

### 5. ROC-AUC (Area Under Receiver Operating Characteristic Curve)

#### The Visual Explanation / The Trade-off
Imagine we don't know what probability threshold to use (0.3? 0.5? 0.8?) to classify a point. Instead of guessing, we test *every single possible threshold* from 0.0 to 1.0. At each step, we record:
*   **True Positive Rate (Sensitivity / Recall)**: Catching the real targets. We want this high (Y-axis).
*   **False Positive Rate (Fall-out)**: Setting off false alarms. We want this low (X-axis).

As we lower the threshold to catch more true targets, we *will* mathematically get more false alarms. The ROC curve just graphs this inescapable trade-off. 

![Receiver Operating Characteristic (ROC) Curve](https://upload.wikimedia.org/wikipedia/commons/1/13/Roc_curve.svg)

#### Definition Summary
*   **Explanation**: Plots the True Positive Rate vs False Positive Rate across all possible probability thresholds.
*   **AUC = 1.0**: Perfect Classifier (Curve perfectly hugs the top-left corner, meaning 100% precision and 100% recall concurrently).
*   **AUC = 0.5**: Useless Random Guessing (Following the straight diagonal line).
*   **When to use**: This is the absolute best metric for comparing *two entirely different models* (e.g., Logistic Regression vs Random Forest) because it scores the model's underlying predictive intelligence *regardless* of the applied 0.5 threshold.

---

## 10. Class Imbalance (In-Depth)

When one class dramatically outweighs another (e.g., Credit Card Fraud is 0.1% of data).

### Solutions
1.  **Don't look at Accuracy**: Look at Recall and F1.
2.  **Class Weights**: Penalize model 1000x geometrically harder for crossing the boundary margin of a minority sample (`class_weight='balanced'`).
3.  **Oversampling (SMOTE)**: Creates mathematically robust **synthetic** new minority data points floating directly on the geometric axis lines connecting adjacent original minority vectors.
4.  **Undersampling**: Truncates majority vectors (Risks throwing away spatial anchors).

---

## 11. Multiclass Logistic Regression

Standard Logistic Regression is binary (it strictly separates Space A from Space B). What if our target has more than 2 classes? (e.g., predicting animal type: Cat, Dog, or Bird). We have two main geometric strategies to adapt our linear hyperplanes for multiple regions.

### 1. One-vs-Rest (OvR) or One-vs-All (OvA)
*   **Intuition**: We break the complex multiclass problem down into multiple independent binary problems.
*   **How it works**: For 3 classes (Cat, Dog, Bird), we train 3 completely separate logistic regression models (drawing 3 separate hyperplanes):
    *   Plane 1: Separates Cat (+1) vs Not a Cat (-1)
    *   Plane 2: Separates Dog (+1) vs Not a Dog (-1)
    *   Plane 3: Separates Bird (+1) vs Not a Bird (-1)
*   **Prediction**: We run a new data point through all 3 planes. Whichever plane evaluates the point with the highest positive geometric distance confidence score "wins" the classification.

### 2. Multinomial Logistic Regression (Softmax Regression)
*   **Intuition**: Instead of training disjointed independent models that might geometrically overlap or create blind spots, we train one unified model that evaluates all planes simultaneously and outputs a mathematically balanced probability distribution.
*   **The Softmax Function**: This is the universal generalization of the Sigmoid curve for multiple dimensions.
    $$ \text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} $$
    *   $z_i$: The raw distance score for the specific class $i$.
    *   $K$: The total number of possible classes (e.g., 3).
    *   $\sum_{j=1}^{K} e^{z_j}$: The sum of the exponentiated scores for *all* classes.

#### 🎈 Softmax Example (How the Math Works)
Imagine our model outputs three raw linear distances ($z$) for an image:
*   Cat ($z_1$) = 2.0 (Strong confidence)
*   Dog ($z_2$) = 1.0 (Weak confidence)
*   Bird ($z_3$) = 0.1 (Very low confidence)

**Step 1: Exponentiate ($e^z$)** 
We raise $e$ to the power of each score. This serves two vital mathematical purposes: it instantly makes every negative distance strictly positive, and it massively exaggerates the differences between high and low scores.
*   Cat: $e^{2.0} \approx 7.39$
*   Dog: $e^{1.0} \approx 2.72$
*   Bird: $e^{0.1} \approx 1.11$

**Step 2: Calculate the Denominator (Total Sum)**
*   Sum = $7.39 + 2.72 + 1.11 = 11.22$

**Step 3: Divide to get Probabilities**
Dividing each exponentiated value by the total sum forces the final array to perfectly evaluate to exactly $1.0$ (100%).
*   Probability of Cat: $7.39 / 11.22 \approx \mathbf{66\%}$
*   Probability of Dog: $2.72 / 11.22 \approx \mathbf{24\%}$
*   Probability of Bird: $1.11 / 11.22 \approx \mathbf{10\%}$

**Prediction**: You get a final probability vector: `[Cat: 66%, Dog: 24%, Bird: 10%]`. You confidently predict **Cat**.

---

## 12. When to Use / Not Use Logistic Regression

### Pros (When to use)
1.  **Geometric Linearity Interpretability**: You explicitly need algebraic log-odds parameter reasoning explaining precisely why an independent transaction flagged.
2.  **Probabilities**: Need confidence score percentage thresholds, not just blanket threshold labels.
3.  **Baseline Efficiency**: Incredibly computationally cheap vector processing; serves universally as an initial control model baseline.

### Cons (When NOT to use)
1.  **Non-Linearity Requirements**: Fails rapidly if topological distributions emulate concentric circles needing tracing curvature logic (Ensemble Trees or Neural Networks fundamentally dominate here).
2.  **Complex Interactions**: Demands extensive explicit feature engineering structure manipulations to map conditional spatial interdependencies. 

---

## 13. Key Takeaways

1.  **Logistic Regression** builds separating geometric hyperplanes to explicitly partition binary categorical data structures.
2.  Equation inherently derived isolating **Linear Functional Margins**, replacing static 0-1 step loss with the smoothed continuous **Logistic Log-Loss Proxy Envelope**.
3.  The raw spatial vector distance output parameter coincidentally perfectly defines the exact statistical **Log-Odds** event manifestation.
4.  Model optimization resolves infallibly into global optimum orientations exploiting rigorous **Convex Hessian Geometry**.
5.  Standard **Accuracy** metric falls fundamentally apart evaluating Imbalanced domain streams; shift evaluation rigor strictly towards calculating **Precision, Recall, F1, and ROC-AUC integrations.**
