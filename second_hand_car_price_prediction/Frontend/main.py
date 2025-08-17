import streamlit as st
import pickle
import cloudpickle
import dill
import pandas as pd
from datetime import datetime
import warnings
import ast

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide", page_title="Second hand car price predictor")

st.title("Second hand car price predictor")

with open("./second_hand_car_price_prediction/Frontend/deps/insurance_values.pkl",'rb') as f:
    insurance_values=pickle.load(f)
    
with open("./second_hand_car_price_prediction/Frontend/deps/fuel_types.pkl",'rb') as f:
    Fuel_types=pickle.load(f)
    
with open("./second_hand_car_price_prediction/Frontend/deps/transmission_values.pkl",'rb') as f:
    transmission_values=pickle.load(f)
    
with open("./second_hand_car_price_prediction/Frontend/deps/companies.pkl",'rb') as f:
    companies=pickle.load(f)
    
with open("./second_hand_car_price_prediction/Frontend/deps/car_models.pkl",'rb') as f:
    models=pickle.load(f)
    
with open("./second_hand_car_price_prediction/Frontend/deps/add_features.pkl",'rb') as f:
    additional_features=pickle.load(f)
    
with open("./second_hand_car_price_prediction/Frontend/deps/rto.pkl",'rb') as f:
    rtos=pickle.load(f)


cols=st.columns(4)

all_years=[year for year in range(int(datetime(1990,1,1).year),int(datetime(2026,1,1).year))]

with cols[0]:
    company_name=st.selectbox("Company",companies)
    insurance=st.selectbox("Insurance",insurance_values)
    seats=st.number_input("No of Seats",step=1)
    Registration_year=st.selectbox("Year of Registraion",all_years)

    
with cols[1]:
    model=st.selectbox("Model",models)
    fuel_type=st.selectbox("Fuel_Type",Fuel_types)
    engine_displacement=st.number_input("Engine Displacement",min_value=0.0,step=0.1)
    manifacture_year=st.selectbox("Year of Manifacture",all_years)

with cols[2]:
    rto=st.selectbox("RTO",rtos)
    transmission=st.selectbox("Transmission",transmission_values)
    kms_driven=st.number_input("Kms driven",step=100,min_value=0)
    # st.write(type(kms_driven))
    power_in_bhp=st.number_input("Power in Bhp",value=0.00,step=0.1,min_value=0.00)

with cols[3]:
    new_vehical_price=st.number_input("New vehical price(in lacs)",min_value=0.0,step=0.1)
    mileage=st.number_input("Mileage in km/l or km/kg",min_value=0.0,step=0.1)
    n_owners=st.number_input("No of owners",min_value=1,step=1)
    additional_features=st.multiselect("Additional Features",additional_features)
    

df=pd.DataFrame({
    'Insurance ':insurance,
    'Mileage in kmpl or km/kg':mileage,
    'Power in bhp':power_in_bhp,
    'new_vehical_price_in_lakh_inr':new_vehical_price,
    'Engine Displacement in cc':engine_displacement,
    'Kms Driven ':int(kms_driven),
    'company_name':company_name,
    'model_detail':model,
    'RTO ':rto,
    'other_features':str(additional_features),
    'Fuel Type ':fuel_type,
    'Transmission ':transmission,
    'Registration Year ':Registration_year,
    'Year of Manufacture ' :manifacture_year,
    'Seats ':seats,
    'No of owners':n_owners
},index=[0])


predict=st.button("Predict")
if predict:
    with open("./second_hand_car_price_prediction/Frontend/deps/ppl.pkl",'rb') as f:
        ppl=cloudpickle.load(f)
        
    tranformed_X=ppl.predict(df)
    
    if tranformed_X[0]<1:
        st.success(f"Predicted second hand value is {round(tranformed_X[0]*100,2)} Thousand")
    elif tranformed_X[0]<100:
        st.success(f"Predicted second hand value is {round(tranformed_X[0],2)} Lakh")
    else:
        st.success(f"Predicted second hand value is {round(tranformed_X[0]/100,2)} Crore")
    