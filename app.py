import streamlit as st
import pandas as pd
import joblib
import os

@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load(os.path.join(os.path.dirname(__file__), "fraud_model.pkl"))
        scaler = joblib.load(os.path.join(os.path.dirname(__file__), "scaler.pkl"))
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.error("Please run fraud_detection_simple.py first to train and save the model.")
        return None, None

model, scaler = load_model_and_scaler()

st.title(" Credit Card Fraud Detector")
st.write("Upload a CSV file of transactions and the app will predict fraud.")

with st.expander("Expected CSV Format"):
    st.write("""
    Your CSV should contain the following columns:
    - Time, V1, V2, ..., V28, Amount
    - Optional: Class (if you want to compare predictions with actual labels)
    
    The V1-V28 columns are PCA-transformed features from the original credit card dataset.
    """)

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None and model is not None and scaler is not None:
    try:
        data = pd.read_csv(uploaded_file)
        
        st.write("### Original Data")
        st.write(f"Shape: {data.shape}")
        st.write(data.head())
        
        features = data.drop("Class", axis=1, errors="ignore")
        
        required_columns = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
        missing_columns = [col for col in required_columns if col not in features.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            st.stop()
        
        features_scaled = features.copy()
        
        features_scaled[['Time', 'Amount']] = scaler.transform(features_scaled[['Time', 'Amount']])
        
        predictions = model.predict(features_scaled)
        prediction_proba = model.predict_proba(features_scaled)
        
        results_df = data.copy()
        results_df["Fraud_Prediction"] = predictions
        results_df["Fraud_Probability"] = prediction_proba[:, 1]  # Probability of fraud
        
        st.write("### Results with Fraud Predictions")
        
        def highlight_fraud(row):
            if row['Fraud_Prediction'] == 1:
                return ['background-color: #ffcccc'] * len(row)  
            else:
                return ['background-color: #ccffcc'] * len(row) 
        
        st.dataframe(
            results_df.head(20).style.apply(highlight_fraud, axis=1),
            use_container_width=True
        )
        
        fraud_count = (predictions == 1).sum()
        total_transactions = len(predictions)
        fraud_percentage = (fraud_count / total_transactions) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Transactions", total_transactions)
        
        with col2:
            st.metric("Suspected Fraud", fraud_count)
            
        with col3:
            st.metric("Fraud Rate", f"{fraud_percentage:.2f}%")
        
        if fraud_count > 0:
            st.warning(f" Detected {fraud_count} suspected fraudulent transactions ({fraud_percentage:.2f}%)")
            
            fraud_transactions = results_df[results_df['Fraud_Prediction'] == 1].copy()
            fraud_transactions = fraud_transactions.sort_values('Fraud_Probability', ascending=False)
            
            st.write("### High-Risk Transactions")
            st.dataframe(
                fraud_transactions[['Time', 'Amount', 'Fraud_Probability']].head(10),
                use_container_width=True
            )
        else:
            st.success(" No fraudulent transactions detected!")
        
        if 'Class' in data.columns:
            actual_fraud = data['Class'].sum()
            accuracy = (predictions == data['Class']).mean()
            
            st.write("### Model Performance (vs Ground Truth)")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Actual Fraud Cases", actual_fraud)
            
            with col2:
                st.metric("Accuracy", f"{accuracy:.2%}")
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="fraud_detection_results.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.write("Please make sure your CSV file has the correct format.")

elif model is None or scaler is None:
    st.info("Please run the training script first to generate the model files.")
else:
    st.info("Please upload a CSV file to get started.")
