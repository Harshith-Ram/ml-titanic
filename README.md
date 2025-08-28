This project demonstrates **Machine Learning Quality Assurance (ML QA)** by validating a predictive model for the Titanic survival dataset using **PyTest** and **DeepChecks**. The goal is to combine **SDET skills with ML workflows** to ensure data and model quality.

---

## 📂 Project Structure
ml-qa-titanic/
├── data/ # Raw Kaggle dataset
│ ├── train.csv
│ ├── test.csv
│ └── gender_submission.csv
├── notebooks/ # EDA + model training notebooks
│ ├── 01_exploration.ipynb
│ └── 02_training.ipynb
├── tests/ # Automated QA (PyTest + DeepChecks)
│ ├── test_model.py
│ └── test_deepchecks.py
├── model/ # Trained model & artifacts
│ ├── titanic_model.pkl
│ └── titanic_data.pkl
├── requirements.txt
└── README.md


---

## ⚡ Features

- **Data QA:** Checks for missing values, feature consistency, and integrity of the dataset.
- **Model QA:** Evaluates ML model performance using accuracy, overfitting detection, and standard evaluation metrics.
- **Automation:** PyTest scripts to validate dataset and model automatically.
- **DeepChecks Integration:** Generates HTML reports for data integrity and model evaluation.
- **Resume/Portfolio Ready:** Demonstrates practical ML QA skills for SDET → ML QA transition.

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt

