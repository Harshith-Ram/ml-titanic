import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load data
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_df = pd.read_csv(os.path.join(project_root, "data/train.csv"))
test_df = pd.read_csv(os.path.join(project_root, "data/test.csv"))

# Preprocessing function
def preprocess(df):
    df = df.copy()
    # Fill numeric missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    # Fill categorical missing values
    df['Embarked'].fillna('S', inplace=True)
    # Encode categorical features
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    # Return only numeric columns used by model
    return df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

X_train = preprocess(train_df)
y_train = train_df['Survived']
X_test = preprocess(test_df)

# Fill missing values just for model training
X_train['Age'].fillna(X_train['Age'].median(), inplace=True)
X_test['Age'].fillna(X_train['Age'].median(), inplace=True)
X_train['Fare'].fillna(X_train['Fare'].median(), inplace=True)
X_test['Fare'].fillna(X_train['Fare'].median(), inplace=True)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save artifacts
os.makedirs(os.path.join(project_root, "model"), exist_ok=True)
joblib.dump(model, os.path.join(project_root, "model/titanic_model.pkl"))
joblib.dump((X_train, X_test, y_train), os.path.join(project_root, "model/titanic_data.pkl"))

print("Model training complete and artifacts saved.")

