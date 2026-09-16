# Customer Purchase Predictor 🛒

Predict whether a customer is likely to purchase a product based on age, estimated salary, and gender, with a Streamlit app for interactive predictions.

## 🛠️ Tools & Tech Stack

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Pandas](https://img.shields.io/badge/Pandas-DataFrame-150458)
![NumPy](https://img.shields.io/badge/NumPy-Array-013243)
![scikit--learn](https://img.shields.io/badge/scikit--learn-Pipeline%20%7C%20SVM%20%7C%20DecisionTree-F7931E)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Plotting-11557C)
![Seaborn](https://img.shields.io/badge/Seaborn-EDA-4C72B0)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B)
![Joblib](https://img.shields.io/badge/Joblib-Model%20Serialization-8A2BE2)

## 📌 Overview

This project analyzes social network ad data to understand which customers are likely to make a purchase, and builds a classification model to predict purchase behavior. It includes:

- Exploratory Data Analysis (EDA) to surface key purchase drivers
- A preprocessing + modeling pipeline (Decision Tree vs. SVM)
- A deployed Streamlit app for real-time, single-customer predictions

## 📊 Dataset

The dataset (`Social_Network_Ads.csv`) contains customer records with the following fields:

| Feature | Description |
|---|---|
| `User ID` | Unique customer identifier (dropped before modeling) |
| `Gender` | Customer gender |
| `Age` | Customer age |
| `EstimatedSalary` | Customer's estimated salary |
| `Purchased` | Target — whether the customer purchased (1) or not (0) |

## 🔍 Exploratory Data Analysis

Key steps performed in the notebook:
- Dropped the non-predictive `User ID` column
- Checked shape, data types, and summary statistics
- Removed duplicate records and checked for missing values
- Visualized distributions of `Age` and `EstimatedSalary`
- Compared `Age` and `EstimatedSalary` against purchase outcome using scatter plots and box plots
- Correlation heatmap across numerical features

## ⚙️ Preprocessing & Modeling Pipeline

Built using `scikit-learn`'s `ColumnTransformer` and `Pipeline`:

- **Numerical features** (`Age`, `EstimatedSalary`) → scaled with `StandardScaler`
- **`Gender`** → one-hot encoded (`handle_unknown="ignore"`)

Two models were trained and compared:

| Model | Notes |
|---|---|
| Decision Tree Classifier | Baseline tree-based model |
| Support Vector Classifier (SVC) | RBF kernel |

Evaluated using **Accuracy, Precision, Recall, and F1 Score**, with the final model pipeline serialized using `joblib` (`model.pkl`).

## 🖥️ Streamlit App

`app.py` loads the trained model pipeline and provides a simple UI to input a customer's age, estimated salary, and gender, returning a prediction of whether that customer is likely to purchase.

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/customer-purchase-predictor.git
cd customer-purchase-predictor
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

### 3. Train the model (optional — a pretrained `model.pkl` can be used instead)
Run through `Customer-Purchase-Predictor.ipynb` to reproduce the EDA and train the model.

### 4. Run the app
```bash
streamlit run app.py
```

## 📂 Project Structure

```
├── Customer-Purchase-Predictor.ipynb   # EDA, preprocessing, model training & evaluation
├── app.py                              # Streamlit app for predictions
├── model.pkl                           # Serialized trained model pipeline
└── README.md
```

## 📈 Possible Improvements

- Hyperparameter tuning (GridSearchCV) for the SVC and Decision Tree models
- Try additional classifiers (Logistic Regression, Random Forest) for comparison
- Add decision boundary visualization for the 2D feature space
- Deploy the app (Streamlit Community Cloud / Docker)

## 📝 License

This project is open-sourced for educational purposes.
