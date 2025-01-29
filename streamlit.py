# importing sources
import streamlit as st
import pickle
import numpy as np
import requests
import pandas as pd
import io
import traceback
# openweather
def get_weather_data(location):
    api_key = "9311f6a18572d18e580c0f9885106d74"  
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    if response.status_code == 200:
        return data['main']['temp']  
    else:
        return None

#Gathering information From Trined and stored  data
try:
    with open("/mount/src/niruthi_project/model_file.pkl", "rb") as f:
        model, label_encoders = pickle.load(f)
except Exception as e:
    print("An error occurred:")
    traceback.print_exc()

#title
st.title("Farmer Advisory System")

#inputs of user
location = st.text_input("Enter Location (like -----> Delhi, Gujarat)")
crop_stage = st.selectbox("Select Crop Stage", label_encoders['Crop Stage'].classes_)
cat_event = st.selectbox("Select Category Event", label_encoders['Any Cat Event'].classes_)

#Temperature
temperature = None
if location:
    temperature = get_weather_data(location)
    if temperature is not None:
        st.write(f"Current Temperature in {location}: {temperature}C")
    else:
        st.write(f"Could not fetch weather data for {location}. Please check the location or try again.")
#sumbit button
if st.button("Get<>Advisory"):
    if temperature is not None:
        
        crop_stage_encoded = label_encoders['Crop Stage'].transform([crop_stage])[0]
        cat_event_encoded = label_encoders['Any Cat Event'].transform([cat_event])[0]

        
        input_features = np.array([[crop_stage_encoded, cat_event_encoded]])

        prediction = model.predict(input_features)
        advisory = label_encoders['Agro Advisory'].inverse_transform(prediction)[0]
#  preparing data for Farmerpreparing data from
        st.write(f"Advisory: {advisory}")

        advisory_df = pd.DataFrame({
            "Location": [location],
            "Crop Stage": [crop_stage],
            "Category Event": [cat_event],
            "Temperature (°C)": [temperature],
            "Advisory": [advisory]
        })

        csv = advisory_df.to_csv(index=False)
        json = advisory_df.to_json(orient="records", lines=True)

#csv download
        st.download_button(
            label="Download Advisory as CSV",
            data=csv,
            file_name="farmer_advisory.csv",
            mime="text/csv"
        )
# json download
        st.download_button(
            label="Download Advisory as JSON",
            data=json,
            file_name="farmer_advisory.json",
            mime="application/json"
        )

    else:
        st.write("Unable to fetch temperature. Please try again.")
