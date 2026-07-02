# Diabetes Prediction with Logistic Regression

This project uses Logistic Regression to predict whether a patient has diabetes based on diagnostic medical measurements. The goal is to apply the machine learning workflow learned in class to a real-world healthcare dataset.

## Author
**Mahmoud Najmeh**  
<img src="https://avatars.githubusercontent.com/u/78208459?u=c3f9c7d6b49fc9726c5ea8bce260656bcb9654b3&v=4" width="200px" style="border-radius: 50%;">

------------------------------------------------------------------------

## Project Overview

The model is trained on the diabetes dataset, which contains medical predictor variables such as glucose level, blood pressure, BMI, insulin level, age, and other diagnostic measurements.

The target variable is `Outcome`:

* `0` = Patient does not have diabetes
* `1` = Patient has diabetes

## Dataset Features

The dataset includes the following columns:

| Feature                  | Description                  |
| ------------------------ | ---------------------------- |
| Pregnancies              | Number of pregnancies        |
| Glucose                  | Plasma glucose concentration |
| BloodPressure            | Diastolic blood pressure     |
| SkinThickness            | Triceps skinfold thickness   |
| Insulin                  | 2-hour serum insulin         |
| BMI                      | Body Mass Index              |
| DiabetesPedigreeFunction | Diabetes pedigree function   |
| Age                      | Patient age                  |
| Outcome                  | Target variable              |

## Technologies Used

* Python
* pandas
* scikit-learn
* uv

## Installation

Clone the repository:

```bash
git clone https://github.com/mahmoudnajmeh/ml-diabetes-prediction.git
cd diabetes-prediction
```

Install dependencies using `uv`:

```bash
uv add pandas scikit-learn
```

## Project Structure

```text
diabetes-prediction/
├── data/
│   └── diabetes.csv
├── diabetes_prediction.py
├── README.md
├── pyproject.toml
└── .gitignore
```

## How to Run

Run the script with:

```bash
uv run python diabetes_prediction.py
```

## Machine Learning Workflow

The script follows these steps:

1. Load the diabetes dataset into a pandas DataFrame.
2. Separate the dataset into features `X` and target variable `y`.
3. Split the data into training and testing sets using a 75/25 split.
4. Scale the feature values using `StandardScaler`.
5. Train a Logistic Regression model.
6. Make predictions on the test data.
7. Print the accuracy score and confusion matrix.
8. Display the number of False Negatives.
9. Print a short reflection about False Negatives and False Positives.

## Example Output

```text
Accuracy Score: 0.6771

Confusion Matrix:
[[36 35]
 [27 94]]

False Negatives (FN): 27
```

## Reflection

In this model, a False Negative means that a patient actually has diabetes, but the model predicts that they do not. This can be dangerous because it may delay diagnosis, treatment, and necessary lifestyle changes.

A False Positive means that a patient does not have diabetes, but the model predicts that they do. This may cause stress and unnecessary medical testing, but it is generally less harmful than a False Negative because further medical tests can confirm the diagnosis.

## GitHub Notes

This repository should not include virtual environments or unnecessary generated files. The `.gitignore` file should exclude files such as:

```text
.venv/
__pycache__/
*.pyc
```

