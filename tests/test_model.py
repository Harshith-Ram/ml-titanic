import pytest
from sklearn.metrics import accuracy_score
import joblib
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model and data
model = joblib.load(os.path.join(project_root, "model/titanic_model.pkl"))
X_train, X_test, y_train = joblib.load(os.path.join(project_root, "model/titanic_data.pkl"))

def test_model_accuracy():
    y_pred = model.predict(X_train)
    acc = accuracy_score(y_train, y_pred)
    assert acc > 0.75, f"Model accuracy too low: {acc}"

def test_no_missing_values():
    assert X_train.isnull().sum().sum() == 0, "Training set contains missing values"

