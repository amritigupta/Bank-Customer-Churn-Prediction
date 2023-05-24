import streamlit as st
import pickle
import numpy as np
import base64
import json
from PIL import Image



#Importing model and label encoders

with open(
        "churn_model.pickle",
        'rb') as f:
    __model = pickle.load(f)

with open("columns (1).json", 'r') as obj:
    __data_columns = json.load(obj)['data_columns']
    __geography = __data_columns[8:11]
    __gender = __data_columns[11:]

def predict_churn(CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary):
    try:
        geography_index = __data_columns.index(Geography.lower())
        gender_index = __data_columns.index(Gender.lower())
    except ValueError as e:
        geography_index = -1
        gender_index = -1

    lis = np.zeros(len(__data_columns))
    lis[0] = CreditScore
    lis[1] = Age
    lis[2] = Tenure
    lis[3] = Balance
    lis[4] = NumOfProducts
    lis[5] = HasCrCard
    lis[6] = IsActiveMember
    lis[7] = EstimatedSalary

    if geography_index >= 0:
        lis[geography_index] = 1
    else:
        lis[geography_index] = -1

    if gender_index >= 0:
        lis[gender_index] = 1
    else:
        lis[gender_index] = -1

    a = np.array(lis).reshape(1, -1)

    churn = __model.predict(a)

    return churn


def main():
    st.title("Prediction of churn customers")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:yellow;text-align:center;">Churn Classification</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)


    CreditScore = st.slider('Select credit score', 300, 900)

    Geography = st.selectbox('Geography', __geography)

    Gender = st.selectbox("Gender", __gender)

    Age = st.slider("Select Age(in years)", 10, 95)

    Tenure = st.selectbox("Select tenure(in years)", ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10', '11', '12', '13', '14', '15'])

    Balance = st.slider("Select balance", 0.00, 250000.00)

    NumOfProducts = st.selectbox('Select number of products', ['1', '2', '3', '4'])

    HasCrCard = st.selectbox("Has credit card or not.(yes-1,no-0)", ['0', '1'])

    IsActiveMember = st.selectbox("Is an active member or not.(yes-1,no-0)", ['0', '1'])

    EstimatedSalary = st.slider("Select the estimated salary", 0.00, 200000.00)



    if st.button('Predict'):
        output = predict_churn(CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary)
        st.success('The probability of customer being churned is {}'.format(output))
        st.balloons()

if __name__=='__main__':
    main()