# import streamlit as st
# import pandas as pd
# import joblib

# model = joblib.load("fraud_detection_model.pkl")

# st.title("Fraud Detection And Prediction App")

# st.markdown("Enter the transaction details to predict whether it is fraudulent or not by using prediction button")

# st.divider()

# transaction_type = st.selectbox("Select Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"])
# amount = st.number_input("Enter Transaction Amount", min_value=0.0, value = 1000.0)
# oldbalanceOrig = st.number_input("Enter Old Balance(Sender)", min_value=0.0, value = 10000.0)
# newbalanceOrig = st.number_input("Enter New Balance(Sender)", min_value=0.0, value = 9000.0)
# oldbalancedest = st.number_input("Enter Old Balance(Receiver)", min_value=0.0, value = 0.0)
# newbalancedest = st.number_input("Enter New Balance(Receiver)", min_value=0.0, value = 0.0)

# if st.button("Predict"):
#     input_data = pd.DataFrame({
#         "type": [transaction_type],
#         "amount": [amount],
#         "oldbalanceOrg": [oldbalanceOrig],
#         "newbalanceOrig": [newbalanceOrig],
#         "oldbalanceDest": [oldbalancedest],
#         "newbalanceDest": [newbalancedest]
#     })
    
#     prediction = model.predict(input_data)[0]
#     st.subheader(f"Prediction : '{int(prediction)}'")
    
#     if prediction == 1:
#         st.error("The transaction is predicted to be fraudulent.")
#     else:
#         st.success("The transaction is predicted to be legitimate.")

import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("fraud_detection_model.pkl")

# Page config
st.set_page_config(
    page_title="Fraud Detection App",
    page_icon="💳",
    layout="centered"
)

# Title
st.markdown(
    "<h1 style='text-align: center;'>💳 Fraud Detection & Prediction App</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; font-size:18px;'>Enter transaction details to check whether it is <b>Fraudulent</b> or <b>Legitimate</b></p>",
    unsafe_allow_html=True
)

st.divider()

# Layout using columns
col1, col2 = st.columns(2)

with col1:
    transaction_type = st.selectbox(
        "🔄 Transaction Type",
        ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"]
    )
    amount = st.number_input(
        "💰 Transaction Amount",
        min_value=0.0,
        value=1000.0
    )
    oldbalanceOrig = st.number_input(
        "🏦 Old Balance (Sender)",
        min_value=0.0,
        value=10000.0
    )

with col2:
    newbalanceOrig = st.number_input(
        "🏦 New Balance (Sender)",
        min_value=0.0,
        value=9000.0
    )
    oldbalancedest = st.number_input(
        "👤 Old Balance (Receiver)",
        min_value=0.0,
        value=0.0
    )
    newbalancedest = st.number_input(
        "👤 New Balance (Receiver)",
        min_value=0.0,
        value=0.0
    )

st.divider()

# Predict button (centered)
predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
with predict_col2:
    predict = st.button("🔍 Predict Fraud", use_container_width=True)

# Prediction
if predict:
    input_data = pd.DataFrame({
        "type": [transaction_type],
        "amount": [amount],
        "oldbalanceOrg": [oldbalanceOrig],
        "newbalanceOrig": [newbalanceOrig],
        "oldbalanceDest": [oldbalancedest],
        "newbalanceDest": [newbalancedest]
    })

    prediction = model.predict(input_data)[0]

    st.divider()
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("🚨 **Fraudulent Transaction Detected!**")
        st.markdown(
            "<div style='background-color:#7f1d1d; padding:15px; border-radius:10px;'>"
            "<b>⚠️ Warning:</b> This transaction shows strong fraud indicators.</div>",
            unsafe_allow_html=True
        )
    else:
        st.success("✅ **Legitimate Transaction**")
        st.markdown(
            "<div style='background-color:#1e3a2f; padding:15px; border-radius:10px;'>"
            "<b>✔ Safe:</b> This transaction appears normal.</div>",
            unsafe_allow_html=True
        )

# Sidebar
st.sidebar.title("ℹ️ About App")
st.sidebar.info(
    """
    This application uses a **Machine Learning model**
    to detect fraudulent financial transactions.

    **Algorithm Used:**
    - Logistic Regression

    **Tech Stack:**
    - Python
    - Pandas
    - Scikit-learn
    - Streamlit
    """
)


