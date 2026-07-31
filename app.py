import streamlit as st
import pickle
import numpy as np
import pandas as pd

# 1. Page Configuration
st.set_page_config(
    page_title="Loan Prediction App",
    page_icon="🏦",
    layout="centered"
)

# 2. Safe Custom CSS (Adapts to Light/Dark Mode)
st.markdown("""
    <style>
    /* Gradient Header */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 25px;
        border-radius: 10px;
        text-align: center;
        color: white !important;
        margin-bottom: 25px;
    }
    
    /* Result Cards with strict text colors for readability */
    .result-card-approved {
        background-color: #d4edda !important;
        color: #155724 !important;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        border: 1px solid #c3e6cb;
    }
    .result-card-rejected {
        background-color: #f8d7da !important;
        color: #721c24 !important;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        border: 1px solid #f5c6cb;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Load Model (Cached for performance)
@st.cache_resource
def load_model():
    return pickle.load(open('loan_app_30Jun.pkl', 'rb'))

model = load_model()

# 4. App Header
st.markdown(
    '<div class="main-header">'
    '<h1 style="color: white; margin:0;">🏦 Loan Approval Prediction</h1>'
    '<p style="margin:0; font-size:18px;">Enter applicant details below to check eligibility</p>'
    '</div>', 
    unsafe_allow_html=True
)

st.markdown("### 📋 Applicant Details")

# 5. User Inputs
col1, col2 = st.columns(2, gap="large")

with col1:
    dependents = st.slider('Dependents', min_value=0, max_value=3)
    app_income = st.number_input('Applicant Income ($)', min_value=1025, max_value=32541, step=100)
    co_app_income = st.number_input('Co-Applicant Income ($)', min_value=0, max_value=9000, step=100)
    loan_amount = st.number_input('Loan Amount (in thousands)', min_value=30, max_value=500, step=10)
    loan_amount_term = st.selectbox('Loan Amount Term (in months)', [84, 120, 180, 240, 300, 360, 480], index=5)
    credit_history = st.selectbox('Credit History (1=Good, 0=Bad)', [0, 1], index=1)

with col2:
    gender = st.selectbox('Gender', ['Male', 'Female'])
    married = st.selectbox('Married', ['Yes', 'No'])
    education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
    self_emp = st.selectbox('Self Employed', ['Yes', 'No'])
    property_type = st.selectbox('Property Type', ['Rural', 'Urban', 'Semi-Urban'])

st.divider()

# 6. Data Preprocessing (Cleaned up logic)
gen_m = 1 if gender == "Male" else 0
mar_yes = 1 if married == "Yes" else 0
edu_grad = 1 if education == "Graduate" else 0
se_yes = 1 if self_emp == "Yes" else 0

prop_surb = 1 if property_type == "Semi-Urban" else 0
prop_urb = 1 if property_type == "Urban" else 0
prop_rur = 1 if property_type == "Rural" else 0

# Format the array
test_data = np.array([gen_m, mar_yes, dependents, edu_grad, se_yes, app_income,
                      co_app_income, loan_amount, loan_amount_term, credit_history,
                      prop_surb, prop_urb]).reshape((1, 12))

test_df = pd.DataFrame(test_data, columns=['Gender', 'Married', 'Dependents', 'Education', 
        'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
        'Loan_Amount_Term', 'Credit_History', 'Property_Area_Semiurban',
        'Property_Area_Urban'])

with st.expander("🔍 View Input Summary"):
    st.dataframe(test_df, use_container_width=True)

# 7. Prediction & Output
st.markdown("<br>", unsafe_allow_html=True)

# Using Streamlit's native primary button for perfect theme compatibility
gen_prediction = st.button("🔮 Predict Loan Status", type="primary", use_container_width=True)

if gen_prediction:
    prediction = model.predict(test_df)[0]
    
    # Check your model's exact output for an approval. Standard is 1 or 'Y'
    if prediction == 1 or prediction == 'Y':
        st.markdown(
            '<div class="result-card-approved">🎉 Congratulations! Your Loan is APPROVED.</div>', 
            unsafe_allow_html=True
        )
        st.balloons()
    else:
        st.markdown(
            '<div class="result-card-rejected">⚠️ Sorry! Your Loan is REJECTED.</div>', 
            unsafe_allow_html=True
        )