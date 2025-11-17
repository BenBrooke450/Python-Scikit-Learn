
## **What `train_test_split` Does**

`sklearn.model_selection.train_test_split` is a **utility function** used to split datasets into **training and testing subsets** (or more generally, multiple subsets).

* The **training set** is used to **fit/train** your machine learning model.
* The **testing set** is used to **evaluate** the model’s performance on unseen data.

Splitting your data is **essential** to avoid overfitting and to get a reliable estimate of how your model will perform on new data.

---

## **Why We Use It**

* **Prevent data leakage**: ensures the model is tested on data it has **never seen** during training.
* **Evaluate performance**: compute metrics like accuracy, MSE, F1-score, etc., on unseen data.
* **Optional validation sets**: you can split the data into train/validation/test for hyperparameter tuning.

---

## **Function Signature**

```python
sklearn.model_selection.train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)
```

### **Parameters**

| Parameter      | Description                                                                                                                                                                    |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `*arrays`      | One or more arrays (lists, NumPy arrays, pandas DataFrames/Series) of the same length. Usually `X` (features) and `y` (targets).                                               |
| `test_size`    | Float, int, or None. <br>- If float, proportion of dataset to include in the test split (e.g., 0.2 = 20%). <br>- If int, absolute number of test samples. <br>- Default: None. |
| `train_size`   | Float, int, or None. Size of the training set. If None, automatically set as complement of `test_size`.                                                                        |
| `random_state` | Controls the **randomness** of the split for reproducibility. Any integer seed will do.                                                                                        |
| `shuffle`      | Boolean, whether to **shuffle the data before splitting**. Default=True.                                                                                                       |
| `stratify`     | Array-like. If not None, data is split **in a stratified fashion**, preserving class proportions (useful for imbalanced datasets).                                             |

---

### **Return Values**

* Returns a **tuple** containing the split arrays:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
```

* Shapes:

  * `X_train` → `(n_train_samples, n_features)`
  * `X_test` → `(n_test_samples, n_features)`
  * `y_train` → `(n_train_samples,)`
  * `y_test` → `(n_test_samples,)`

> If more than 2 arrays are provided, all arrays are split **consistently** along the first axis.

---

## **Examples**

### **Example 1: Simple split**

```python
from sklearn.model_selection import train_test_split
import numpy as np

X = np.arange(10).reshape((5, 2))  # 5 samples, 2 features
y = np.array([0, 1, 0, 1, 0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

print("X_train:\n", X_train)
print("X_test:\n", X_test)
print("y_train:", y_train)
print("y_test:", y_test)
```

* `test_size=0.4` → 40% of the data in test set (2 samples out of 5).
* `random_state=42` → ensures the split is **reproducible**.

---

### **Example 2: Stratified split for classification**

```python
X = np.arange(20).reshape((10, 2))
y = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # balanced classes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

print("y_train:", y_train)
print("y_test:", y_test)
```

* `stratify=y` ensures **class distribution is preserved** in both train and test sets.
* Useful when dealing with **imbalanced datasets**.

---

### **Example 3: Splitting more than two arrays**

```python
X = np.arange(12).reshape((6, 2))
y = np.array([0, 1, 0, 1, 0, 1])
sample_weights = np.array([1, 2, 1, 2, 1, 2])

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, sample_weights, test_size=0.33, random_state=0
)
```

* All arrays are **split consistently**, preserving the correspondence between features, targets, and weights.

---

## **Best Practices**

1. **Always shuffle your data** unless it has a natural order (e.g., time series).
2. **Set `random_state`** for reproducibility.
3. **Use `stratify=y`** for classification with imbalanced classes.
4. **Decide test size carefully**:

   * Typical splits: 70/30, 80/20, 75/25
   * Smaller datasets may require more careful cross-validation.
5. For **time-series data**, do **not shuffle** — use a sequential split instead.

---

## ** Summary Table**

| Feature        | Description                                                                                    |
| -------------- | ---------------------------------------------------------------------------------------------- |
| Purpose        | Split dataset into training and testing sets                                                   |
| Inputs         | X (features), y (targets), optional extra arrays                                               |
| Key Parameters | test_size, train_size, random_state, shuffle, stratify                                         |
| Outputs        | X_train, X_test, y_train, y_test (and other arrays if provided)                                |
| Notes          | Shuffle by default, stratify preserves class proportions, random_state ensures reproducibility |


