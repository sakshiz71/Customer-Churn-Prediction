import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")

# Load model file (customer_churn_model.pkl)
@st.cache_resource
def load_artifacts():
    with open("customer_churn_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    loaded_model = model_data["model"]
    features_names = model_data["features_names"]

    with open("encoder.pkl", "rb") as f:
        encoders = pickle.load(f)

    return loaded_model, encoders, features_names

loaded_model, encoders, features_names = load_artifacts()

#Title
st.title("📊 Customer Churn Predictor")
st.write("Fill in the customer details and click Predict.")

# Input form 
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 1)

with col2:
    st.subheader("Services")
    phone = st.selectbox("Phone Service", ["No", "Yes"])
    multi = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

with col3:
    st.subheader("Billing")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 29.85)
    total = st.number_input("Total Charges ($)", 0.0, 10000.0, 29.85)

# Predict button 
if st.button("🔍 Predict Churn", use_container_width=True):

    # Build dataframe exactly like your notebook
    customer_data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multi,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }
    customer_data_df = pd.DataFrame([customer_data])

    # Encode 
    for column, encoder in encoders.items():
        customer_data_df[column] = encoder.transform(customer_data_df[column])

    # Predict
    prediction = loaded_model.predict(customer_data_df)
    pred_prob = loaded_model.predict_proba(customer_data_df)

    # result
    st.divider()
    if prediction[0] == 1:
        st.error("⚠️ Prediction: This customer will CHURN")
        st.metric("Churn Probability", f"{pred_prob[0][1]*100:.1f}%")
    else:
        st.success("✅ Prediction: This customer will NOT churn")
        st.metric("Retention Probability", f"{pred_prob[0][0]*100:.1f}%")

    st.progress(float(pred_prob[0][1]))
    st.caption(f"Churn risk: {pred_prob[0][1]*100:.1f}%  |  No churn: {pred_prob[0][0]*100:.1f}%")