Sure! Here’s a **full, detailed summary of `accuracy_score`** in scikit-learn, including what it is, how it works, its formula, parameters, and examples.

---

## **1️⃣ What `accuracy_score` Is**

`accuracy_score` is a **metric in scikit-learn** used to evaluate the performance of a **classification model**.

* It measures the **proportion of correct predictions** out of all predictions made.
* It is commonly used for **binary classification**, **multi-class classification**, and **multi-label classification**.

Mathematically:

$$
\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}
$$

or equivalently:

$$
\text{Accuracy} = \frac{\sum_{i=1}^{n} \mathbf{1}{y_i = \hat{y}_i}}{n}
$$

Where:

$$
* $$(y_i)$$ = true label of sample (i)
* $$(\hat{y}_i)$$ = predicted label of sample (i)
* $$(\mathbf{1}{y_i = \hat{y}_i})$$ = indicator function (1 if true, 0 if false)
* (n) = total number of samples


---

## **Key Points About Accuracy**

1. **Simple and intuitive**: tells the fraction of correct predictions.
2. **Range**: always between 0 and 1 (or 0% to 100% if multiplied by 100).
3. **Sensitive to class imbalance**:

   * If one class dominates, a high accuracy may be misleading.
   * Example: 95% of samples are class A → predicting all A gives 95% accuracy but poor performance on minority class.

---

## ** Parameters of `accuracy_score` in scikit-learn**

```python
sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
```

| Parameter       | Description                                                                                                             |
| --------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `y_true`        | Array of true labels (ground truth)                                                                                     |
| `y_pred`        | Array of predicted labels from the model                                                                                |
| `normalize`     | If `True` (default), returns fraction of correct predictions; if `False`, returns the **number of correct predictions** |
| `sample_weight` | Optional array of weights for each sample; can give more importance to certain samples                                  |

---

## ** Examples**

### **Example 1: Binary Classification**

```python
from sklearn.metrics import accuracy_score

y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

acc = accuracy_score(y_true, y_pred)
print("Accuracy:", acc)
```

**Output:**

```
Accuracy: 0.8
```

* 4 out of 5 predictions are correct → 0.8 (80%).

---

### **Example 2: Multi-Class Classification**

```python
y_true = [0, 2, 1, 2, 0]
y_pred = [0, 1, 1, 2, 0]

acc = accuracy_score(y_true, y_pred)
print("Accuracy:", acc)
```

**Output:**

```
Accuracy: 0.8
```

* Correct predictions: index 0, 2, and 3 → 4 out of 5 → 80%.

---

### **Example 3: Returning Number of Correct Predictions**

```python
acc_count = accuracy_score(y_true, y_pred, normalize=False)
print("Number of correct predictions:", acc_count)
```

**Output:**

```
Number of correct predictions: 4
```

* Instead of fraction, we get **count** of correct predictions.

---

### ** Advantages**

* Very **easy to interpret** and compute.
* Works for **binary, multi-class, and multi-label** problems.

### ** Limitations**

* **Not suitable for imbalanced datasets** — a high accuracy can be misleading if one class dominates.
* Does not distinguish **types of errors** (false positives vs false negatives).
* For imbalanced datasets, metrics like **precision, recall, F1-score, or balanced accuracy** are often better.

---

### **Summary**

* `accuracy_score` measures the **proportion of correct predictions**:
  [
  \text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}
  ]
* Parameters: `y_true`, `y_pred`, optional `normalize` and `sample_weight`.
* Advantages: simple, intuitive, works for multi-class classification.
* Limitations: sensitive to class imbalance, does not capture types of errors.
* Typical usage:

```python
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_true, y_pred)
```

