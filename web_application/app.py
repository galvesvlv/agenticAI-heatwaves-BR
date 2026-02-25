# app.py -> web_application

import streamlit as st
import requests
import base64
from datetime import date

API_URL = "http://api:8001/forecast"

st.set_page_config(layout="wide")

st.title("Heatwave Forecast System")

# Input
selected_date = st.date_input(
                              "Select forecast month",
                              value=date(2026, 1, 1)
                              )

if st.button("Run Forecast"):

    with st.spinner("Running forecast..."):

        response = requests.post(
                                 API_URL,
                                 json={"target_time": str(selected_date)}
                                 )

        if response.status_code == 200:
            data = response.json()

            # Report
            st.subheader("Scientific Report")
            st.write(data["report"])

            # Image
            st.subheader("Prediction Map")

            image_bytes = base64.b64decode(data["image_base64"])
            st.image(image_bytes)

        else:
            st.error("Error contacting API")
