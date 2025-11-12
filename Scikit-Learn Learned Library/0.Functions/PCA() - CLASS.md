
## **What `sklearn.decomposition.PCA` Does**

The `PCA` class in **scikit-learn** performs **Principal Component Analysis**, which is a technique to reduce the dimensionality of data while retaining the directions of maximum variance.

Mathematically, PCA:

1. Computes the **covariance matrix** of the data (or uses SVD directly).
2. Finds the **eigenvectors and eigenvalues** of the covariance matrix.
3. Sorts eigenvectors by decreasing eigenvalue to select the most important principal components.
4. Projects the original data onto these components, producing a lower-dimensional representation.

---

<br>

## **Key Parameters in `PCA`**

| Parameter      | Description                                                                                                                                                          |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `n_components` | Number of principal components to keep. Can be an integer (number of PCs), float between 0 and 1 (fraction of variance to keep), or `'mle'` for automatic selection. |
| `copy`         | If True (default), the original data is copied before performing PCA.                                                                                                |
| `whiten`       | If True, the components are scaled so that the resulting features have unit variance. Often used for preprocessing before machine learning.                          |
| `svd_solver`   | Algorithm for computing the SVD. Options: `'auto'`, `'full'`, `'arpack'`, `'randomized'`.                                                                            |
| `random_state` | Controls randomness for stochastic SVD solvers.                                                                                                                      |

---

<br>

## **Important Attributes After Fitting PCA**

After calling `fit` or `fit_transform`:

| Attribute                   | Description                                                                                                |
| --------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `components_`               | Principal axes in feature space (each row = one principal component). Shape = `(n_components, n_features)` |
| `explained_variance_`       | Eigenvalues of the covariance matrix. Variance explained by each component.                                |
| `explained_variance_ratio_` | Fraction of total variance explained by each principal component.                                          |
| `singular_values_`          | Singular values corresponding to each principal component.                                                 |
| `mean_`                     | Per-feature empirical mean, used for centering the data before projection.                                 |
| `n_features_`               | Number of features in the data.                                                                            |
| `n_samples_`                | Number of samples in the data.                                                                             |

---

<br>

## ** Typical Workflow**

### **Step 1: Import and initialize PCA**

```python
from sklearn.decomposition import PCA

# Keep 2 principal components
pca = PCA(n_components=2)
```

### **Step 2: Fit PCA to data**

```python
X = [[2.5, 2.4],
     [0.5, 0.7],
     [2.2, 2.9],
     [1.9, 2.2],
     [3.1, 3.0],
     [2.3, 2.7],
     [2.0, 1.6],
     [1.0, 1.1],
     [1.5, 1.6],
     [1.1, 0.9]]

pca.fit(X)
```

---

### **Step 3: Inspect Results**

```python
# Principal components
print(pca.components_)
# Variance explained
print(pca.explained_variance_)
# Ratio of variance explained
print(pca.explained_variance_ratio_)
```

* `components_` gives the **directions** of maximum variance.
* `explained_variance_ratio_` tells you **how much information each component retains**.

---

### **Step 4: Transform Data**

To get the data projected onto the principal components:

```python
X_pca = pca.transform(X)
print(X_pca)
```

* `X_pca` now has shape `(n_samples, n_components)` — the **lower-dimensional representation** of the original data.

---

<br>

### ** Example: Reduce 3D data to 2D**

```python
import numpy as np
from sklearn.decomposition import PCA

# 3D dataset
X = np.array([[2.5, 2.4, 0.5],
              [0.5, 0.7, 0.2],
              [2.2, 2.9, 0.8],
              [1.9, 2.2, 0.7],
              [3.1, 3.0, 0.9],
              [2.3, 2.7, 0.6],
              [2.0, 1.6, 0.4],
              [1.0, 1.1, 0.3],
              [1.5, 1.6, 0.5],
              [1.1, 0.9, 0.2]])

# Reduce to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Principal components:\n", pca.components_)
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Projected data:\n", X_pca)
```

* This example reduces the original 3-feature dataset into **2 principal components**.
* The projected data `X_pca` can then be used for visualization or as input to another ML model.

---

<br>

### ** Notes / Best Practices**

1. **Center the data**: PCA in scikit-learn automatically centers the data by subtracting the mean.
2. **Scaling**: If features have different units, standardize them (e.g., using `StandardScaler`) before PCA.
3. **Choosing `n_components`**:

   * Use a fixed number of components if you know the dimensionality you want.
   * Or, choose a fraction of variance to retain:

     ```python
     pca = PCA(n_components=0.95)  # keep 95% of variance
     ```
4. **Whitening**: If you plan to use PCA for machine learning preprocessing, you can use `whiten=True` to decorrelate components.

---

### **Summary**

* `sklearn.decomposition.PCA` performs dimensionality reduction by computing principal components via SVD.
* `components_` = directions of maximum variance, `explained_variance_ratio_` = importance of each component.
* You fit PCA with `.fit()` and get transformed, lower-dimensional data with `.transform()` or combined `.fit_transform()`.
* Common use cases: visualization, feature reduction, noise removal, and preprocessing for ML models.

