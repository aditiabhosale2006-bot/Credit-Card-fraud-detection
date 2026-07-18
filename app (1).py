import streamlit as st
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="centered"
)

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("💳 Credit Card Fraud Detection")
st.markdown("Enter transaction details to predict whether it is **fraudulent or legitimate**.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount (₹)", min_value=0.0, max_value=10000.0, value=100.0, step=0.01)
    hour = st.slider("Transaction Hour", min_value=0, max_value=23, value=12)
    distance_from_home = st.number_input("Distance from Home (km)", min_value=0.0, max_value=500.0, value=5.0, step=0.1)
    velocity_last_hour = st.slider("Transactions in Last Hour", min_value=0, max_value=10, value=1)

with col2:
    merchant_category = st.selectbox("Merchant Category", ["grocery", "online_shopping", "travel", "entertainment", "restaurant", "other"])
    is_online = st.radio("Online Transaction?", ["No", "Yes"])
    is_foreign = st.radio("Foreign Transaction?", ["No", "Yes"])

st.markdown("---")

is_online_val = 1 if is_online == "Yes" else 0
is_foreign_val = 1 if is_foreign == "Yes" else 0

categories = ["entertainment", "grocery", "online_shopping", "restaurant", "travel"]
cat_encoded = [1 if merchant_category == c else 0 for c in categories]

input_data = np.array([[amount, hour, distance_from_home, is_online_val, is_foreign_val, velocity_last_hour] + cat_encoded])

if st.button("🔍 Detect Fraud", use_container_width=True):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.error(f"🚨 **FRAUDULENT Transaction Detected!**  Confidence: {probability[1]*100:.1f}%")
    else:
        st.success(f"✅ **Legitimate Transaction.**  Confidence: {probability[0]*100:.1f}%")

    col3, col4 = st.columns(2)
    with col3:
        st.write(f"- Amount: **₹{amount}**")
        st.write(f"- Hour: **{hour}:00**")
        st.write(f"- Distance from home: **{distance_from_home} km**")
        st.write(f"- Transactions last hour: **{velocity_last_hour}**")
    with col4:
        st.write(f"- Merchant: **{merchant_category}**")
        st.write(f"- Online: **{is_online}**")
        st.write(f"- Foreign: **{is_foreign}**")
        st.write(f"- Fraud probability: **{probability[1]*100:.1f}%**")

st.markdown("---")
st.caption("Built by Aditi Bhosale | Random Forest Classifier | Credit Card Fraud Detection")
