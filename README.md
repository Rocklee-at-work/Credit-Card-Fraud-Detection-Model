# Credit Card Fraud Detection

This project builds and evaluates a deep learning model to detect fraudulent credit card transactions from a highly imbalanced dataset. The primary goal is to maximize the detection of fraudulent cases (Recall) while maintaining high overall accuracy.

This repository includes:
* `CreditCard_Fraud_Model_Final.ipynb`: The Jupyter Notebook containing all data preprocessing, model training, and evaluation.
* `README.md`: This summary file.

## The Dataset

* **Source:** [Kaggle's Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-lab/creditcardfraud)
* **Contents:** The dataset contains 284,807 transactions that occurred in September 2013 by European cardholders.
* **Imbalance:** It is highly imbalanced, with only **492 fraudulent transactions (0.172%)**.
* **Features:** Features `V1` through `V28` are the result of a PCA transformation. The only features that have not been transformed are `Time` and `Amount`.

---

## Project Methodology

1.  **Data Preprocessing:**
    * The `Amount` and `Time` features were scaled using `StandardScaler` to ensure all features have a similar distribution, which is crucial for model performance.

2.  **Handling Class Imbalance (SMOTE):**
    * Given the severe class imbalance, the **SMOTE (Synthetic Minority Over-sampling Technique)** was applied **only to the training data**.
    * SMOTE creates new, synthetic examples of the minority (fraud) class, allowing the model to learn its patterns more effectively without simply guessing the majority (non-fraud) class.

3.  **Model Experimentation:**
    * Several models were trained and evaluated to find the best performer, including:
        * Logistic Regression
        * Random Forest
        * XGBoost
        * Isolation Forest
        * Deep Neural Network (DNN)

4.  **Model Selection:**
    * After comparing performance metrics across all models, the **Deep Neural Network (DNN)** was selected as the best-performing model due to its superior balance of recall and precision for this specific problem.

---

## Final Model: Deep Neural Network (DNN) Results

The final DNN model demonstrates a strong ability to identify fraudulent transactions while minimizing false positives.

### Performance Metrics

* **Accuracy:** 0.9984 (99.84%)
    * *The model correctly classified 99.84% of all transactions.*
* **Precision:** 0.5287 (52.87%)
    * *When the model predicted a transaction was fraudulent, it was correct 52.87% of the time.*
* **Recall:** 0.8469 (84.69%)
    * *The model successfully identified **84.69%** of all actual fraudulent transactions.*
* **F1-Score:** 0.6510
    * *The harmonic mean of Precision and Recall, providing a balanced measure.*
* **ROC AUC:** 0.9688
    * *Indicates an excellent capability for the model to distinguish between fraudulent and non-fraudulent transactions.*

### Confusion Matrix

This matrix breaks down the model's predictions on the test set:

| | **Predicted: Not Fraud** | **Predicted: Fraud** |
| :--- | :--- | :--- |
| **Actual: Not Fraud** | **56,790** (True Negative) | **74** (False Positive) |
| **Actual: Fraud** | **15** (False Negative) | **83** (True Positive) |

---

## Conclusion

The model performs extremely well in a challenging, imbalanced environment. The **high Recall (84.69%)** is the most important success, as it means the model caught 83 out of 98 fraudulent transactions in the test set, successfully protecting against most fraud.

The **Precision (52.87%)** shows a clear trade-off: to catch this many fraudulent cases, the model also flagged 74 legitimate transactions as fraudulent (false positives). In a real-world scenario, this is often an acceptable trade-off, as reviewing a small number of false positives is preferable to missing a large amount of fraud. The Deep Neural Network provided the best balance for this critical task.
