import streamlit as st
import pandas as pd
import numpy as np
import pickle

# lr = pickle.load(open('lr.pkl','rb'))
dt = pickle.load(open('dt.pkl','rb'))
rf = pickle.load(open('rf.pkl','rb'))
# knn = pickle.load(open('knn.pkl','rb'))


model = st.sidebar.selectbox('Select the Model',['Decision Tree','Random Forest',
                                                 'Logistic Reg'])   #'KNN_Classifier'])


st.header('Railway Failure Type Prediction')

col1,col2,col3 = st.columns(3)

with col1:
    region = st.selectbox('Region',['Central Railway', 'Southern Railway', 'South Central Railway',
       'Eastern Railway', 'Northern Railway', 'Western Railway'])
    season = st.selectbox('Season',['Summer','Winter','Monsson'])
    train_type = st.selectbox('Train Type',['Passenger','Metro','Express','Freight'])
    train_age_yrs =  st.slider('Train Age',1,34,5)
    avg_speed = st.slider('Avg Speed',20.0,160.0,70.0)
    dist = st.slider('Dist Travelled(km)',5000,2000000,125000)
    amb_temp = st.slider('Ambient Temp',-0.4,58.2,27.8)
    humidity_per = st.slider('Humidity Percent',10.0,100.0,62.8)

with col2:
   rainfall = st.slider('Rainfall_mm',0.0,287.8,15.0)
   wheel_wear_per = st.slider('Wheel_wear_per',0.0,287.8,15.0)
   track_vibration_level = st.slider('Track Vibration Level',0.0,8.96,4.4)
   rail_wear_mm = st.slider('Rail Wear',0.380,25.0,9.42)
   bearing_temp = st.slider('Bearing temp',28.80,217.44,76.0)
   axle_temp =  st.slider('Axle temp',27.90,169.920,64.10)
   brake_pad_wear_per = st.slider('Brake Pad Wear Per',0.0,100.0,41.7)
   brake_pressure_psi = st.slider('Brake Pressure PSI',39.90,140.0,96.7)

with col3:
    battery_voltage =  st.slider('Battery Volatge',17.38,30.67,24.41)
    last_maintenance_days =  st.slider('last_maintenance_days',1.0,364.0,182.3)
    sensor_health_index =  st.slider('sensor_health_index',0.1,100.0,63.37)
    inspection_score = st.slider('inspection_score',40.0,100.0,64.86)
    delay_minutes = st.slider('delay_minutes',0.0,160.0,12.95)
    failure_type = st.selectbox('failure_type',['Wheel Defect','Bearing Failure','Track Defect','Track Failure','Signal Failure'])
    maintenance_required = st.selectbox('maintenance_required',[0,1],0)
    risk_score = st.slider('risk_score',30.0,100.0,71.4)


# Target variable - Failure Severity

test = [region,season,train_type,train_age_yrs,avg_speed,dist,amb_temp,
        humidity_per,rainfall,wheel_wear_per,track_vibration_level,rail_wear_mm,
        bearing_temp,axle_temp,brake_pad_wear_per,brake_pressure_psi,
        battery_voltage,last_maintenance_days,sensor_health_index,inspection_score,
        delay_minutes,failure_type,maintenance_required,risk_score]

test_data = np.array(test).reshape(1,24)
test_df = pd.DataFrame(test_data,columns=['region', 'season', 'train_type', 'train_age_years',
       'average_speed_kmph', 'distance_travelled_km', 'ambient_temperature_c',
       'humidity_percent', 'rainfall_mm', 'wheel_wear_percent',
       'track_vibration_level', 'rail_wear_mm', 'bearing_temperature_c',
       'axle_temperature_c', 'brake_pad_wear_percent', 'brake_pressure_psi',
       'battery_voltage', 'last_maintenance_days', 'sensor_health_index',
       'inspection_score', 'delay_minutes', 'failure_type',
       'maintenance_required', 'risk_score'])

st.write(test_df)

predict_button = st.button('Predict Failure Severity')

if predict_button:
    if model == 'Decision Tree':
        st.success(dt.predict(test_df)[0])
    elif model == 'Random Forest':
        st.success(rf.predict(test_df)[0])
    #elif model == 'Logistic Reg':
     #   st.success(lr.predict(test_df)[0])

