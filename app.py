import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Purchase Prediction",
    layout='centered'
)
model=pickle.load(open('model.pkl','rb'))
scaler=pickle.load(open('scaler.pkl','rb'))
feature_columns=pickle.load(open('feature_columns.pkl','rb'))

st.title('Social network ads Prediction')
st.subheader("Purchase prediction")
st.write(
    'Enter Customer details to predict'
    'whether they are likely to Purchase or not'
)
gender=st.selectbox("select gender",['Male','Female'])

age = st.number_input(
    "Enter Age",
    min_value=18,
    max_value=100,
    step=1
)

salary = st.number_input(
    "Enter Estimated Salary (₹)",
    min_value=0,
    max_value=1000000,
    value=50000,
    step=1000
) 

if st.button("Predict Purchase", type="primary"):
    input_data = pd.DataFrame({
        'Age':[age],
        'Gender':[gender],
        "EstimatedSalary":[salary]
    })
    #onehot encoding
    input_data = pd.get_dummies(
        input_data,
        columns = ['Gender'],dtype = int
    )
     # Match the training columns and their order
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Scale input data
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("The customer is likely to purchase the product!")
    else:
        st.info("The customer is unlikely to purchase the product.")
    st.write("Purchase Probability: {:.2f}%".format(probability * 100))