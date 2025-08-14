import streamlit as st
import pickle
import pandas as pd

st.set_page_config(layout="wide", page_title="Second hand car price predictor")

st.title("Second hand car price predictor")

with open("../../data/frontend/insurance_values.pkl",'rb') as f:
    insurance_values=pickle.load(f)
    
with open("../../data/frontend/fuel_types.pkl",'rb') as f:
    Fuel_types=pickle.load(f)
    
with open("../../data/frontend/transmission_values.pkl",'rb') as f:
    transmission_values=pickle.load(f)
    
with open("../../data/processed/companies.pkl",'rb') as f:
    companies=pickle.load(f)
    
with open("../../data/processed/car_models.pkl",'rb') as f:
    models=pickle.load(f)
    
with open("../../data/processed/add_features.pkl",'rb') as f:
    additional_features=pickle.load(f)
    
with open("../../data/processed/rto.pkl",'rb') as f:
    rtos=pickle.load(f)

cols=st.columns(4)

with cols[0]:
    company_name=st.selectbox("Company",companies)
    insurance=st.selectbox("Insurance",insurance_values)
    seats=st.text_input("No of Seats")
    Registration_year=st.text_input("Year of Registraion")


with cols[1]:
    model=st.selectbox("Model",models)
    fuel_type=st.selectbox("Fuel_Type",Fuel_types)
    engine_displacement=st.text_input("Engine Displacement")
    manifacture_year=st.text_input("Year of Manifacture")

with cols[2]:
    rto=st.selectbox("RTO",rtos)
    transmission=st.selectbox("Transmission",transmission_values)
    kms_driven=st.text_input("Kms driven")
    power_in_bhp=st.text_input("Power in Bhp")

with cols[3]:
    additional_features=st.multiselect("Additional Features",additional_features)
    new_vehical_price=st.text_input("New vehical price")
    mileage=st.text_input("Mileage in km/l or km/kg")
    

df=pd.DataFrame({
    'Insurance':insurance,
    'Mileage in kmpl or km/kg':mileage,
    'Power in bhp':power_in_bhp,
    'new_vehical_price_in_lakh_inr':new_vehical_price,
    'Engine Displacement in cc':engine_displacement,
    'Kms Driven':kms_driven,
    'company':company_name,
    'model':model,
    'rto':rto,
    'add_features':str(additional_features),
    'Fuel Type':fuel_type,
    'Transmission':transmission,
    'Registration Year':Registration_year,
    'Year of Manufacture':manifacture_year,
    'Seats':seats
},index=[0])


predict=st.button("Predict")
print(additional_features)
st.dataframe(pd.read_csv('../../data/frontend/demo.csv'))
if predict:
    df.to_csv('../../data/frontend/demo.csv',index=False)