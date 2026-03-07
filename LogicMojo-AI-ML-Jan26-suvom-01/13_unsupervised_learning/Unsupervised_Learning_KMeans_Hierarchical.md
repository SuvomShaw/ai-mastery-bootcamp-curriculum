# Unsupervised Learning: K-Means & Hierarchical Clustering

---

# 1. What is Unsupervised Learning?

## Definition

Unsupervised learning is a machine learning paradigm where the algorithm learns patterns from **unlabeled data**.

There is **no target variable** (no y). The algorithm tries to discover **hidden structure** inside the data.

Unlike supervised learning:

| Supervised     | Unsupervised      |
| -------------- | ----------------- |
| Input + Output | Only Input        |
| Predict labels | Discover patterns |

Instead of predicting outcomes, it tries to answer questions like:

* Are there **groups** in the data?
* Are some observations **similar**?
* Are there **outliers**?
* Can the data be **compressed** into fewer dimensions?

Formally, given a dataset:

$$
X = \{x_1, x_2, x_3, \dots, x_n\}
$$

There is **no corresponding label** (y). The goal is to discover **structure in X itself**.

---

## Simple Real Life Example

Imagine a shop owner who has **1000 customers** but no labels.

He only has information like:

| Customer | Income | Spending | Age |
| -------- | ------ | -------- | --- |
| A        | 40k    | 80       | 25  |
| B        | 120k   | 20       | 55  |
| C        | 60k    | 65       | 35  |

He wants to answer:

* Which customers behave similarly?
* Which customers should receive premium offers?

Unsupervised learning will **automatically group similar customers together**.

---

### 🧠 Think About It

**Q:** Why can't we use classification here?

### ✅ Answer

Because classification requires **labeled data**.

Example:

| Income | Spending | Label        |
| ------ | -------- | ------------ |
| 40k    | 70       | High Spender |

Here we don't have labels.

---

### 🎯 Follow-up Question

**Q:** If we later manually label clusters, what type of learning can we apply afterward?

**Answer:** **Supervised learning**

---

# 2. Major Types of Unsupervised Learning

| Type                         | Purpose                          |
| ---------------------------- | -------------------------------- |
| **Clustering**               | Group similar data               |
| **Dimensionality Reduction** | Reduce features                  |
| **Association Rules**        | Find relationships between items |

Examples:

| Algorithm               | Type                     |
| ----------------------- | ------------------------ |
| K-Means                 | Clustering               |
| Hierarchical Clustering | Clustering               |
| DBSCAN                  | Clustering               |
| PCA                     | Dimensionality Reduction |
| Apriori                 | Association Rules        |

In these notes we focus on **Clustering**.

---

# 3. What is Clustering?

Clustering means **grouping similar data points together**.

Goal:

* Points in the **same cluster** should be **very similar**
* Points in **different clusters** should be **very different**

Mathematically:

Minimize **intra-cluster distance**
Maximize **inter-cluster distance**

---

### 🧠 Think About It

**Q:** If two points are very close in feature space, should they belong to the same cluster?

### ✅ Answer

Yes. Clustering assumes **similar points should belong to the same cluster**.

Distance is usually measured using:

* Euclidean distance
* Manhattan distance

---

## Visual Intuition

![Image](https://images.openai.com/static-rsc-3/gB4uWHMoZ39xMx2RIUijASzLkDpp7pMaL_O17bokhP1Dof3jc3GvjGcJZzkZrzkqsNMLlB0u1uYnCsQtslMMbJ7DdzDxrgkvl1l8p3o0JUY?purpose=fullsize\&v=1)

![Image](https://images.openai.com/static-rsc-3/OC3fWRRLarHMcP1VTQlhW0mUTZ4y80CiBJjT5LvPDX_gWnzYc_aPuikd08614zAq72_Nmkk2-VnbXYhs0HnsF8w43XEkXTBuoEoa1YePzd8?purpose=fullsize\&v=1)

![Image](https://images.openai.com/static-rsc-3/1ZT-g2FSIVOd4leL7oigF-c9aMXJ2G1_X99F7w9dadj2DcKnxx4ejtNzxbsLoqMrQmlC0qrh0MsRjy9g8WczvRDB_I7HzsGIbw1pYxPWHvI?purpose=fullsize\&v=1)

![Image](https://images.openai.com/static-rsc-3/cW8S_8U1UilDx51AU8DzPwxWYG3MMOD2d-7--S0hk2wlkxSKaRi_dnoJZkfZ0AcDjUrV29C_bL5X6vGDfk6XGARSMIBHhlbXqegPIvtLG6A?purpose=fullsize\&v=1)

Points naturally form **groups or clusters**.
The algorithm identifies these clusters automatically.

---

# 4. Distance Metric

Most clustering algorithms use **distance** to measure similarity.

Euclidean distance:

$$
d(x,y) = \sqrt{(x_1-y_1)^2 + (x_2-y_2)^2}
$$

Example:

Points
```
A(2,3)
B(5,7)
```

Distance
$$
\sqrt{(2-5)^2 + (3-7)^2}
$$

---

### 🧠 Quick Question

If the distance between two points is **very large**, what does it mean?

### ✅ Answer

They are **very different** and belong to different clusters.

---

# 5. K-Means Clustering

## Definition

K-Means is a **centroid-based clustering algorithm** that partitions data into **K clusters**.

Each cluster has a **center called centroid**.

The algorithm tries to minimize:

$$
\sum_{i=1}^{K} \sum_{x \in C_i} ||x - \mu_i||^2
$$

Where:

* (C_i) = cluster
* (\mu_i) = centroid
* Distance usually **Euclidean**

Goal:

**Minimize the distance between points and their cluster centroid**

---

### 🧠 Think About It

Why do you think the algorithm is called **K-Means**?

### ✅ Answer

Because:

* **K** = number of clusters
* **Means** = centroid calculated using **mean of points** in the cluster.

---

# 6. Intuition Behind K-Means

Imagine:

A group of **100 people**.

You want to create **3 project groups** based on **skills and interest**.

Steps:

1. Randomly choose **3 people as group leaders** (centroids).
2. Assign every person to the **closest leader**.
3. Recalculate **leader as average of group**.
4. Repeat until groups stop changing.

This is exactly how **K-Means works**.

---

### 🧠 Think About It

Why do we recompute the centroid after assigning points?

### ✅ Answer

Because the centroid must represent the **average location of the cluster**. This process continuously improves the cluster assignment until it stabilizes.

---

# 7. Visual Explanation of K-Means

![Image](https://d2o2utebsixu4k.cloudfront.net/_-%20visual%20selection%20-%202025-05-22T013131315-a1393a2a12bf441d82d9b0f4e4666527.png)

![Image](https://miro.medium.com/v2/resize%3Afit%3A1200/1%2AigBfOi1IFWA_H3aNZG0bzQ.png)

![Image](https://www.researchgate.net/publication/328411264/figure/fig2/AS%3A683977018966027%401540084237543/K-means-clustering-algorithm-The-steps-of-K-means-clustering-algorithm-are-outlined.png)

![Image](https://people.revoledu.com/kardi/tutorial/kMean/image/Algorithm_clip_image002_0002.gif)

Key idea:
* Centroids move
* Clusters adjust
* Eventually stabilize

---

# 8. K-Means Algorithm (Step-by-Step with Example)

Let's trace K-Means with a simple 1D dataset to make distances and calculations easy to follow.
Dataset: $X = \{2, 3, 8, 9\}$

### Step 1: Choose K (Number of clusters)
Let's choose **$K = 2$**.

---

### Step 2: Initialize Centroids
Randomly select $K$ points as our starting centroids. Let's say we pick:
* Centroid 1 ($C_1$) = **2**
* Centroid 2 ($C_2$) = **9**

---

### Step 3: Assign Points to Nearest Centroid
Measure the distance of every point to both centroids.
* **Point 2**: Distance to $C_1$ (2) is 0. Distance to $C_2$ (9) is 7. $\rightarrow$ Assign to **Cluster 1**.
* **Point 3**: Distance to $C_1$ (2) is 1. Distance to $C_2$ (9) is 6. $\rightarrow$ Assign to **Cluster 1**.
* **Point 8**: Distance to $C_1$ (2) is 6. Distance to $C_2$ (9) is 1. $\rightarrow$ Assign to **Cluster 2**.
* **Point 9**: Distance to $C_1$ (2) is 7. Distance to $C_2$ (9) is 0. $\rightarrow$ Assign to **Cluster 2**.

Current Groups:
* Cluster 1: $\{2, 3\}$
* Cluster 2: $\{8, 9\}$

---

### Step 4: Update Centroids
Compute the new center (mean) for each cluster.
* New $C_1$ = Average of $\{2, 3\}$ = $(2+3)/2$ = **2.5**
* New $C_2$ = Average of $\{8, 9\}$ = $(8+9)/2$ = **8.5**

---

### Step 5: Repeat Until Convergence
Reassign points based on the new centroids (2.5 and 8.5):
* **Point 2**: Closer to 2.5 $\rightarrow$ **Cluster 1**
* **Point 3**: Closer to 2.5 $\rightarrow$ **Cluster 1**
* **Point 8**: Closer to 8.5 $\rightarrow$ **Cluster 2**
* **Point 9**: Closer to 8.5 $\rightarrow$ **Cluster 2**

No points changed clusters! The centroids have stabilized.
**Stop.** The algorithm has converged.

---

### 🧠 Think About It

What happens if centroids stop moving?

### ✅ Answer

The algorithm has **converged**. Clusters are stable.

---

### 🎯 Follow-up Question

Does K-Means guarantee a **global optimum solution**?

**Answer:** **No. It may converge to a local optimum depending on initialization.** (This is why we run it multiple times!).

---

# 9. Example: Customer Segmentation

Dataset:

| Income | Spending Score |
| ------ | -------------- |
| 15     | 39             |
| 15     | 81             |
| 16     | 6              |
| 18     | 94             |
| 19     | 72             |

Clusters might become:

| Cluster | Meaning                  |
| ------- | ------------------------ |
| 1       | Low income high spending |
| 2       | High income low spending |
| 3       | Moderate customers       |

---

### 🧠 Think About It

Why is customer clustering useful for businesses?

### ✅ Answer

Because companies can:
* Target ads specific to groups.
* Offer personalized discounts.
* Improve marketing strategy.



# 11. Choosing the Best K (Elbow Method)

Problem:
How do we choose **K**?

We use **WCSS (Within Cluster Sum of Squares)**.

## WCSS Formula

$$
WCSS = \sum_{i=1}^{K} \sum_{x \in C_i} ||x - \mu_i||^2
$$

Goal:
Lower WCSS = tighter, better clusters.

---

## Elbow Method Visualization

![Image](https://miro.medium.com/v2/resize%3Afit%3A1358/format%3Awebp/0%2Anza9_m-68Uoe6DTW)

![Image](https://www.researchgate.net/publication/344027507/figure/fig2/AS%3A931092106321920%401599001065415/Elbow-method-for-optimal-value-of-K.ppm)

![Image](https://www.researchgate.net/publication/355111689/figure/fig2/AS%3A1076757562163201%401633730416286/Elbow-Curve-on-the-basis-of-Within-Cluster-Sum-of-Squares-WCSS-3-k-Mode-Clustering-and.ppm)

Choose K at the **elbow point** (where the rapid drop slows down).

---

### 🧠 Think About It

What happens if K = number of data points?

### ✅ Answer

Each point becomes its own cluster.
WCSS = **0**
But this is **useless clustering** because there's no generalization.

---

# 12. Advantages of K-Means

✔ Very fast
✔ Easy to implement
✔ Works well for large datasets
✔ Scales easily

---

# 13. Limitations of K-Means

❌ Must choose K beforehand
❌ Sensitive to outliers
❌ Assumes spherical clusters
❌ Different initialization → different results

---

### 🧠 Think About It

Why is K-Means sensitive to outliers?

### ✅ Answer

Because the centroid is the **mean** of points in the cluster, and the mean is heavily affected by extreme values.
Example:
`Cluster points: 10, 11, 12`
`Outlier: 100` -> Centroid shifts drastically!

---

# 14. What is Hierarchical Clustering?

Hierarchical clustering builds a **tree of clusters**.

Instead of specifying K first, we build a **hierarchy** and choose clusters later.

The tree structure is called a **Dendrogram**.

---

### 🧠 Think About It

Why might hierarchical clustering be useful compared to K-Means?

### ✅ Answer

Because we may not know **how many clusters exist** in the data beforehand.
Hierarchical clustering allows us to visually explore the structure and decide K later.

---

# 15. Dendrogram Visualization

![Image](https://www.researchgate.net/publication/48194320/figure/fig1/AS%3A307395533262848%401450300214331/Example-of-a-dendrogram-from-hierarchical-clustering.png)

![Image](https://miro.medium.com/0%2A0gop9RK2hvVlcBBR.png)

Important idea:
Cut the tree **horizontally** to get clusters.

---

### 🧠 Think About It

If a horizontal line intersects **4 vertical branches** on the dendrogram, how many clusters are there?

### ✅ Answer

**4 clusters.**

---

# 16. Types of Hierarchical Clustering

## 1. Agglomerative (Bottom-Up)

**Definition:** The algorithm starts by treating each individual data point as a single cluster and then successfully merges the two closest clusters until all points have been merged into a single, massive root cluster.

Start with:
`N clusters` (each point is its own cluster)

Merge until:
`1 cluster` remains (the root of the tree).

*Most common approach in machine learning libraries like scikit-learn.*

---

### 🧠 Think About It

Wait, what is the point of merging everything? If in the end we just have **1 giant cluster** with all the data inside it, early groups are lost and we have no useful segments. Why do we go all the way to 1?

### ✅ Answer

This is the most common confusion! We **do not use** the final 1 cluster. 

The goal of the algorithm is not the final cluster, the goal is **the journey (the Dendrogram tree)**.

Think of a **sports tournament bracket (like the World Cup)**:
1. You start with 32 individual teams (N clusters).
2. They play each other, merging into groups of 16, then 8, then 4.
3. Finally, there is 1 winner (the 1 cluster).

If I give you just the winner, that's useless for grouping.
But if I give you the **entire printed bracket**, you can look at the "Final 4" or "Elite 8" and say *"Ah, these teams are the most similar/competitive groups."*

By forcing the algorithm to merge all the way to 1, it is forced to draw the **entire tree (dendrogram) from bottom to top**.
Once the tree is drawn, the algorithm stops.
**Then, WE (the humans) look at the tree and "cut" it horizontally at the level that gives us the best groups.** We are keeping a record of every single merge so we can rewind time and pick the perfect spot.

---

## 2. Divisive (Top-Down)

**Definition:** The exact opposite of Agglomerative. The algorithm begins with all data points in one single cluster. At each step, it splits the cluster into two smaller clusters based on maximum distance until each point is in its own individual cluster.

Start with:
`1 cluster` (all points grouped together)

Split repeatedly until:
`N clusters` (each point is isolated).

*This approach is rarely used in standard machine learning because deciding exactly where/how to split a large cluster is computationally much more difficult than just finding the two closest points to merge.*

---

# 17. Agglomerative Clustering Steps (With Example)

Let's use a 1D dataset: $X = \{1, 4, 5, 9\}$

### Step 1: Start with N clusters
Each point is its own independent cluster.
`Clusters: {1}, {4}, {5}, {9}`

---

### Step 2: Merge the closest clusters
The distance between 4 and 5 is only 1. They are the closest pair.
*Merge {4} and {5}.*
`Clusters: {1}, {4, 5}, {9}`

---

### Step 3: Merge again
Measure distances between the remaining clusters. The distance between 1 and {4, 5} is smaller than the distance between {4, 5} and 9.
*Merge {1} and {4, 5}.*
`Clusters: {1, 4, 5}, {9}`

---

### Step 4: Final Merge
Only two clusters left. Merge them.
`Clusters: {1, 4, 5, 9}`

Notice how we built a tree (dendrogram) from the bottom up! Each step tracks a horizontal line you can "cut" later.

---

# 18. Divisive Clustering Steps (With Example)

Let's trace the exact same dataset top-down: $X = \{1, 4, 5, 9\}$

### Step 1: Start with 1 giant cluster
All points are grouped together.
`Cluster: {1, 4, 5, 9}`

---

### Step 2: Split the cluster into two
We look for the largest gap in the data to make the most sensible cut. The biggest gap is between 5 and 9 (a distance of 4).
*Split {1, 4, 5} and {9}.*
`Clusters: {1, 4, 5}, {9}`

---

### Step 3: Split again
Now we look at the cluster {1, 4, 5}. The biggest remaining gap here is between 1 and 4 (a distance of 3).
*Split {1} and {4, 5}.*
`Clusters: {1}, {4, 5}, {9}`

---

### Step 4: Final Split
Split the remaining group {4, 5}.
`Clusters: {1}, {4}, {5}, {9}`

Notice how this is the exact reverse of Agglomerative! We work top-down, splitting based on maximum distance.

---

### 🧠 Think About It

Why is Agglomerative called **bottom-up clustering**?

### ✅ Answer

Because we start from the "bottom" (individual data points as tiny clusters) and gradually merge them "up" into larger and larger groups.





### 🧠 Think About It

Wait a minute, earlier we said Hierarchical Clustering doesn't need K (number of clusters) and merges until there is only **1 giant cluster**. But in the code above, we put `n_clusters=3`! Why?

### ✅ Answer

This is the difference between **theory** and **software implementation**. 

**In Theory:** We build the full dendrogram up to 1 root cluster, print the picture, look at it visually, and say "Ah, I want to cut this tree at the level that gives me 3 branches."

**In Scikit-Learn:** Building the *entire* tree and holding it in memory for massive datasets is very slow and crashes computers. By telling Scikit-Learn `n_clusters=3`, we are telling the algorithm: *"Go ahead and start merging from the bottom up. But instead of going all the way to 1 single cluster, **stop merging immediately** as soon as you reach exactly 3 clusters."* 

It acts as an automatic "cut" on the dendrogram to save time!

---

# 20. Evaluating Clustering Performance (Metrics)

Since there are no "true labels" (no $y$) in Unsupervised Learning, we cannot use accuracy, precision, or recall. Instead, we use internal metrics that measure **cohesion (how tight a cluster is)** and **separation (how far apart distinct clusters are)**.

### Metric 1: WCSS (Within-Cluster Sum of Squares)

* Also called **Inertia**.
* Measures **cohesion** (tightness).
* It is the sum of squared distances between each point and its cluster's centroid.
* **Goal:** As low as possible. (But remember, if $K=N$, WCSS is $0$, which is completely useless. That is why we use the Elbow Method).

**Visual Example of WCSS (Inertia):**
![WCSS Intuition](https://miro.medium.com/v2/resize:fit:1400/1*rw8IUza1dbffBhiA4i0GNQ.png)
*Notice how WCSS measures the distance from the cluster's center (the red X) to every point inside of it. Tighter circles = Lower WCSS.*

---

### Metric 2: Silhouette Score

* Measures **both cohesion AND separation**.
* Evaluates how similar an object is to its own cluster compared to other clusters.
* Range: **$-1$ to $+1$**.

**Visual Example of Silhouette Score:**
![Silhouette Intuition](https://scikit-learn.org/stable/_images/sphx_glr_plot_kmeans_silhouette_analysis_001.png)
*Left: Poor clustering (scores close to 0 or negative). Right: Good clustering (scores close to +1).*

**How to interpret Silhouette Score:**
* **$+1$ (Excellent):** The point is very close to its own cluster and very far from other clusters. Perfect clustering.
* **$0$ (Borderline):** The point is right on the boundary between two clusters.
* **$-1$ (Terrible):** The point has been placed in the wrong cluster.

*Formula Intuition:*
Let $a$ be the average distance inside its own cluster (cohesion).
Let $b$ be the average distance to the nearest neighboring cluster (separation).
$$ \text{Silhouette Score} = \frac{b - a}{\max(a, b)} $$

---

### 🧠 Think About It

If I trained a K-Means model and my overall Silhouette Score is **-0.5**, what should I do?

### ✅ Answer

A negative score means your points are closer to *other* clusters than they are to their *own* cluster! You have assigned the data terribly. You should immediately:
1. Try a different number of clusters ($K$).
2. Check for extreme outliers.
3. Try standardizing/scaling your data (because distance metrics fail if features aren't on the same scale).

---

# 21. Applications of Clustering

### Biology
Gene similarity / grouping species.

### Document Clustering
Topic grouping for news articles.

### Social Networks
Community detection.

### Marketing
Customer grouping.

### Image Segmentation
Segment regions in images.

---

### 🧠 Think About It

Which clustering algorithm is better for **very large datasets (millions of rows)**?

### ✅ Answer

**K-Means**.
Hierarchical clustering is computationally expensive ($O(N^3)$ or $O(N^2 \log N)$), making it too slow for massive data.

---

# 21. K-Means vs Hierarchical Clustering (Summary Table)

| Feature       | K-Means      | Hierarchical |
| ------------- | ------------ | ------------ |
| Approach      | Partitioning | Tree-based   |
| Speed         | Fast         | Slow         |
| Need K        | Yes          | No           |
| Visualization | Low          | High (Dendrogram) |
| Large data    | Excellent    | Poor         |

---

# 22. High-Level Summary

### K-Means (The Instant Organizer)
Coordinator: "Create **3 groups** of people."
People are assigned immediately and iterate to find the centers.

### Hierarchical (The Relationship Builder)
Coordinator: "First, pair up with your closest neighbor. Then, merge pairs into groups of 4. Then merge groups."
Eventually, the whole group is one giant network.

---

# 23. Real Industry Use Cases

### Netflix
User clustering for recommendation engines.

### Amazon
Customer segmentation based on buying patterns.

### Banking
Fraud detection (flagging anomalies far from clusters).

---

# 24. Final Review

### Question 1
**Q:** Why does K-Means sometimes fail to cluster datasets correctly (like a crescent moon shape)?
**Answer:** Because K-Means assumes clusters are **spherical (circular)** around the centroid.

---

### Question 2
**Q:** In K-Means, what happens if K is too small?
**Answer:** Distinct groups are forced together, losing valuable divisions in the data.

---

### Question 3
**Q:** In K-Means, what happens if K is too large?
**Answer:** Clusters become too fragmented, and patterns lose their meaning/generalization.


