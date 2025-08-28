import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

#Load data
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_df = pd.read_csv(os.path.join(project_root, "data/train.csv"))
test_df = pd.read_csv(os.path.join(project_root, "data/test.csv"))

# Load data
#train_df = pd.read_csv("../data/train.csv")
#test_df = pd.read_csv("../data/test.csv")

# Quick overview
train_df.head()
train_df.info()
train_df.describe()

# Visualizations
sns.countplot(x='Survived', data=train_df)
sns.countplot(x='Pclass', hue='Survived', data=train_df)
sns.countplot(x='Sex', hue='Survived', data=train_df)

# Check missing values
train_df.isnull().sum()

