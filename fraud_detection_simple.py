
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

print(" Starting fraud detection training...")

# Load dataset
data = pd.read_csv("creditcard.csv")
print(f"✅ Dataset loaded: {data.shape}")
print(f"Fraud cases: {data['Class'].sum()} / {len(data)} ({data['Class'].sum()/len(data)*100:.2f}%)")

X = data.drop('Class', axis=1)
y = data['Class']

scaler = StandardScaler()
X[['Time','Amount']] = scaler.fit_transform(X[['Time','Amount']])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"✅ Data balanced: {y_train_resampled.value_counts().to_dict()}")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "XGBoost": XGBClassifier(
        use_label_encoder=False, 
        eval_metric='logloss', 
        random_state=42,
        n_jobs=-1,
        verbosity=0  
    )
}

results = {}
best_model_name = ""
best_roc_auc = 0

for name, model in models.items():
    print(f"⏳ Training {name}...")
    
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    results[name] = {
        "conf_matrix": conf_matrix,
        "roc_auc": roc_auc
    }
    
    if roc_auc > best_roc_auc:
        best_roc_auc = roc_auc
        best_model_name = name
    
    tn, fp, fn, tp = conf_matrix.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"✅ {name} completed:")
    print(f"   ROC AUC: {roc_auc:.4f}")
    print(f"   Fraud Recall: {recall:.4f}")
    print(f"   Fraud Precision: {precision:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print("-" * 40)

print(f"\n WINNER: {best_model_name}")
print(f"Best ROC AUC: {best_roc_auc:.4f}")

best_model = models[best_model_name]
joblib.dump(best_model, "fraud_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print(f"✅ Model saved: fraud_model.pkl")
print(f"✅ Scaler saved: scaler.pkl")
print(f"🚀 Training completed successfully!")

print(f"\n Confusion Matrix for {best_model_name}:")
conf = results[best_model_name]["conf_matrix"]
print(f"True Negatives:  {conf[0,0]:,}")
print(f"False Positives: {conf[0,1]:,}")
print(f"False Negatives: {conf[1,0]:,}")
print(f"True Positives:  {conf[1,1]:,}")

