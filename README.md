# Ames Housing Price Prediction: End-to-End Regression Pipeline

## 📌 Overview
This project is a complete, production-ready machine learning pipeline designed to predict residential property prices in Ames, Iowa. It utilizes the modern Ames Housing dataset (79 explanatory variables) and demonstrates advanced data science workflows, including automated preprocessing, hyperparameter tuning, and model serialization.

## 🛠️ Technology Stack & Techniques
* **Language:** Python
* **Libraries:** Scikit-Learn, Pandas, NumPy, Joblib
* **Algorithm:** Ridge Regression ($L2$ Regularization)
* **Optimization:** Grid Search with 5-Fold Cross-Validation (`GridSearchCV`)
* **Data Processing:** Scikit-Learn `Pipeline` and `ColumnTransformer`

## 🧠 The Pipeline Architecture
To prevent data leakage and ensure reproducible results, all data transformations are bundled into a unified Scikit-Learn Pipeline:
1. **Numerical Features:** Missing values are handled via median imputation, followed by standard scaling (`StandardScaler`) to normalize feature weights.
2. **Categorical Features:** Missing data is treated as a distinct category, and text variables are transformed into a dense mathematical matrix using `OneHotEncoder`.
3. **Estimator:** The cleaned, unified dataset is fed directly into a Ridge Regressor to prevent overfitting on the high-dimensional data created by the One-Hot Encoding.

## 📊 Model Performance
The model's hyperparameter (`alpha`) was optimized using Grid Search, testing values from 0.1 to 500.0. The optimal penalty was mathematically determined to be `alpha=10.0`.

**Final Evaluation Metrics (Test Set):**
* **$R^2$ Score:** 0.8951 (Explains ~89.5% of the variance in house prices)
* **Root Mean Squared Error (RMSE):** $29,001.98

## 💾 Model Serialization
The final, tuned pipeline is exported as a `.pkl` file using `joblib`. This allows the model to be instantly loaded into a separate application to predict new house prices without needing to retrain the algorithm or rebuild the preprocessing steps.
