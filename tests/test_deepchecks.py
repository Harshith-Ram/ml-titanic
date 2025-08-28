import os
import joblib
from deepchecks.tabular.suites import data_integrity, model_evaluation

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model and data
model = joblib.load(os.path.join(project_root, "model/titanic_model.pkl"))
X_train, X_test, y_train = joblib.load(os.path.join(project_root, "model/titanic_data.pkl"))

def test_data_integrity():
    suite = data_integrity()
    results = suite.run(X_train)   # positional argument only
    print("⚠ Data Integrity Results:")
    print(results)

def test_model_evaluation():
    suite = model_evaluation()
    results = suite.run(X_train, model, y_train)  # positional arguments only
    print("✅ Model Evaluation Results:")
    print(results)

