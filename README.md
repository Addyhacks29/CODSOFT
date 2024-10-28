# CODSOFT
All codes of Codsoft Internship are over here


# Machine Learning Models Collection

This repository contains four machine learning projects covering classification, regression, and anomaly detection. Each project focuses on a different dataset and demonstrates unique aspects of data preprocessing, model training, and evaluation. 

## Table of Contents

1. [Titanic Survival Prediction](#titanic-survival-prediction)
2. [Iris Species Classification](#iris-species-classification)
3. [Sales Prediction](#sales-prediction)
4. [Credit Card Fraud Detection](#credit-card-fraud-detection)

---

## 1. Titanic Survival Prediction

Predicts the survival of passengers aboard the Titanic using logistic regression, based on features such as age, gender, family size, fare, embarkation point, and class of service.

### Features
- **Data Preprocessing:** Handles missing values and creates additional features for improved accuracy.
- **Model:** Logistic Regression
- **Prediction Function:** Allows input of passenger attributes for survival prediction.

### Example Usage
```python
print(predict_survival(age=30, sex='male', family_size=1, fare=10.5, embarked='S', pclass=3))
```

### Requirements
- Python 3.x
- pandas
- scikit-learn

---

## 2. Iris Species Classification

Classifies the species of Iris flowers using logistic regression, based on sepal length, sepal width, petal length, and petal width.

### Features
- **Data Preprocessing:** Simple train-test split with no NaN handling, as the dataset is clean.
- **Model:** Logistic Regression
- **Interactive Input:** Allows users to input flower measurements and get a species prediction.

### Example Usage
```python
# Enter the measurements to identify the Iris species:
sepal_length = float(input("Sepal length (cm): "))
sepal_width = float(input("Sepal width (cm): "))
petal_length = float(input("Petal length (cm): "))
petal_width = float(input("Petal width (cm): "))

# Model predicts species based on measurements
predicted_species = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
print(f"\nPredicted Iris species: {predicted_species[0]}")
```

### Requirements
- Python 3.x
- pandas
- scikit-learn

---

## 3. Sales Prediction

Predicts sales figures based on advertising budgets across TV, radio, and newspapers using linear regression.

### Features
- **Data Preprocessing:** Prepares data by selecting relevant features and splitting into training and testing sets.
- **Model:** Linear Regression
- **Prediction Function:** Allows input of advertising budgets to predict sales.

### Example Usage
```python
tv_budget = 100
radio_budget = 50
newspaper_budget = 25
predicted_sales = predict_sales(tv_budget, radio_budget, newspaper_budget)
print(f"Predicted Sales: {predicted_sales}")
```

### Requirements
- Python 3.x
- pandas
- scikit-learn

---

## 4. Credit Card Fraud Detection

Detects fraudulent credit card transactions using a Random Forest classifier, with data preprocessing techniques like SMOTE for handling class imbalance.

### Features
- **Data Preprocessing:** Uses SMOTE for class imbalance and StandardScaler for normalization.
- **Model:** Random Forest Classifier
- **Evaluation Metrics:** Includes precision, recall, and F1-score to assess performance.

### Example Usage
```python
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
```

### Requirements
- Python 3.x
- pandas
- scikit-learn
- imbalanced-learn

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/machine-learning-models.git
   cd machine-learning-models
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## License

This project is open-source and available under the [MIT License](LICENSE).

