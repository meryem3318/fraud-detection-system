# Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

This project implements a machine learning system that detects fraudulent credit card transactions with 97.6% accuracy using advanced algorithms and an interactive web interface built with Streamlit.

### Key Features:
- Three ML models (Logistic Regression, XGBoost, Random Forest)
- Handles imbalanced datasets using SMOTE
- Interactive web app for real-time predictions
- Performance metrics and visualizations

## 📊 Model Performance

| Model | ROC AUC | Precision | Recall | F1-Score |
|-------|---------|-----------|---------|----------|
| Logistic Regression | 0.9698 | 0.0581 | 0.9184 | 0.1094 |
| XGBoost | **0.9760** | 0.3455 | 0.8673 | 0.4942 |
| Random Forest | 0.9723 | 0.2156 | 0.8901 | 0.3456 |

 **Best Model**: XGBoost with 97.60% ROC AUC


## Screenshots

### Web Interface
![Main Interface](screenshots/ss1.png)

### Fraud Detection Results
![Fraud Detection Results](screenshots/ss2.png)

### Model Performance
![Model Comparison](screenshots/ss3.png)

### Dataset Information
- **Source**: Kaggle Credit Card Fraud Detection Dataset
- **Size**: 284,807 transactions, 31 features
- **Imbalance**: 492 fraud cases (0.172%) vs 284,315 normal (99.828%)
- **Features**: Time, Amount, V1-V28 (PCA-transformed)

## License

This project is licensed under the MIT License.