''' 

The Backend Analysis of the Delhi Weather AQI Dataset has been done on the Jupyter Notebook.  

Now, comes the part of the Prediction using the Streamlit and the random Forest Regressor Algorithm.  

This is done to make a proper user interface for the Dataset Analysis which can be easily used to amke the AQI predictions. 
'''  

# Importing the libraries 
import streamlit as st 
import pandas as pd 
import numpy as np 

from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import r2_score 

# Streamlit Page Configuration  

st.set_page_config(page_title = "AQI Predictor", layout = "centered")  

st.title("🌫️Delhi AQI Prediction App")  
st.write("Random Forest Regressor based AQI Predictor")  

# Loading the Data 
@st.cache_data 
def load_data(): 
    df = pd.read_csv("Delhi_AQI_Weather.csv") 
    return df 

df = load_data() 

st.subheader("📊 Dataset Preview") 
st.dataframe(df.head()) 

features = ['lat', 'lon', 'temp_c', 'humidity', 'pressure_mb', 'windspeed_kph', 'pm2_5', 'pm10', 'co', 'no2']  

x = df[features] 
y = df['aqi_index']  

# Train Test and Split the model 

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.20, random_state = 0) 

# scaling the data  
sc= StandardScaler() 
x_train =sc.fit_transform(x_train) 
x_test = sc.transform(x_test) 


# Model Tranining 
model = RandomForestRegressor(n_estimators = 500, random_state = 42) 
regressor = model.fit(x_train, y_train)  

# Model Evaluation 
y_pred = regressor.predict(x_test) 
r2 = r2_score(y_test, y_pred) 


st.success(f"✅ Model R² Score: {r2:.2f}")  

#  Now comes the sidebar inputs  

st.sidebar.header("🔧 Input Parameters")  

lat = st.sidebar.slider("Latitude", 28.4, 28.9,28.6) 
lon = st.sidebar.slider('Longitude', 76.83, 77.40,77.23) 
temp_c = st.sidebar.slider("Temperature(°C)", 0.0, 50.0, 25.0) 
humidity = st.sidebar.slider("Humidity (%)", 10, 100, 50) 
pressure_mb = st.sidebar.slider("Pressure (MB)", 950.0, 1050.0,1010.0) 
windspeed_kph =st.sidebar.slider("Wind Speed:", 0.0,  30.0, 5.0) 

pm2_5 = st.sidebar.slider("PMM2.5", 0.0,500.0, 150.0) 
pm10 = st.sidebar.slider("PM10", 0.0, 5000.0, 1200.0) 
co = st.sidebar.slider("CO", 0.0,5000.0, 1200) 
no2 =st.sidebar.slider("NO2", 0.0, 500.0, 40.0) 

# input the data 
input_data = np.array([[lat, lon,temp_c, humidity,pressure_mb, windspeed_kph, pm2_5, pm10, co, no2]])  

# Scaling the input data 
input_scaled = sc.transform(input_data) 


if st.button("🔮 Predict AQI"): 
    prediction = regressor.predict(input_scaled)[0]  

    st.subheader("📈 Predicted AQI") 
    st.metric("AQI index", f"{int(prediction)}")  

    if prediction <= 50: 
        st.success("Good Air Quality 😊") 

    elif prediction <= 100: 
        st.info("Moderate Air Quality 😐") 

    elif prediction <= 200: 
        st.warning("Poor Air Quality 😷") 

    else: 
        st.error("Severe Air Pollution ☠️") 




