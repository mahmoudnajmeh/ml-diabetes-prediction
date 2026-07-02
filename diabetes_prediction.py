import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATASET_PATH = "data/diabetes.csv"
TARGET_COLUMN = "Outcome"
TEST_SIZE = 0.25
RANDOM_STATE = 42


def load_data(file_path):
    return pd.read_csv(file_path)


def prepare_data(data):
    X = data.drop(TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN]
    return X, y


def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)
    return accuracy, matrix


def print_results(data, X_train, X_test, accuracy, matrix):
    false_negatives = matrix[1, 0]

    print("First 5 rows of the dataset:")
    print(data.head())

    print(f"\nDataset shape: {data.shape}")

    print("\nTarget distribution:")
    print(data[TARGET_COLUMN].value_counts())

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    print(f"\nAccuracy Score: {accuracy:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print(f"\nFalse Negatives (FN): {false_negatives}")

    print("\nReflection")
    print("Question 1:")
    print(
        f"My confusion matrix produced {false_negatives} False Negatives. "
        "These are patients who actually have diabetes but were incorrectly predicted as not having diabetes."
    )

    print("\nQuestion 2:")
    print(
        "A False Negative means a patient who actually has diabetes is incorrectly predicted as not having the disease. "
        "This can delay diagnosis, treatment, and necessary lifestyle changes, increasing the risk of serious complications "
        "such as heart disease, kidney damage, nerve damage, and vision loss."
    )
    print(
        "A False Positive means a patient without diabetes is incorrectly predicted as having the disease. "
        "Although this may cause unnecessary stress and additional medical tests, it is generally less harmful than a False Negative "
        "because the patient can receive further testing to confirm the diagnosis."
    )
    print(
        "Overall, False Negatives are more dangerous because they may prevent patients from receiving timely medical care."
    )


def main():
    diabetes_data = load_data(DATASET_PATH)

    X, y = prepare_data(diabetes_data)

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    model = train_model(X_train_scaled, y_train)

    accuracy, matrix = evaluate_model(model, X_test_scaled, y_test)

    print_results(diabetes_data, X_train, X_test, accuracy, matrix)


if __name__ == "__main__":
    main()