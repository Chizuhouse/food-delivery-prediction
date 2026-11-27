
import streamlit as st
import pandas as pd
import joblib

# Memuat model
model = joblib.load("best_food_delivery_model.pkl")

st.set_page_config(
    page_title="Food Delivery Prediction",
    page_icon="🍔",
    layout="centered"
)

st.title("🍔 Food Delivery Time Prediction")
st.write("Prediksi estimasi waktu pengiriman makanan menggunakan Machine Learning.")

st.divider()

st.subheader("Masukkan Informasi Pengiriman")

col1, col2 = st.columns(2)

with col1:
    distance = st.number_input(
        "Jarak Pengiriman (km)",
        min_value=0.1,
        max_value=50.0,
        value=5.0
    )

    weather = st.selectbox(
        "Cuaca",
        ["Clear", "Rainy", "Snowy", "Windy", "Foggy"]
    )

    traffic = st.selectbox(
        "Tingkat Kemacetan",
        ["Low", "Medium", "High"]
    )

    time_of_day = st.selectbox(
        "Waktu Pengiriman",
        ["Morning", "Afternoon", "Evening", "Night"]
    )

with col2:
    vehicle = st.selectbox(
        "Jenis Kendaraan",
        ["Bike", "Scooter", "Car"]
    )

    preparation = st.number_input(
        "Waktu Persiapan Makanan (menit)",
        min_value=1,
        max_value=120,
        value=15
    )

    experience = st.number_input(
        "Pengalaman Kurir (tahun)",
        min_value=0,
        max_value=50,
        value=2
    )

st.divider()

if st.button("Prediksi Waktu Pengiriman", use_container_width=True):

    weather_traffic = weather + "_" + traffic

    input_data = pd.DataFrame([{
        "Distance_km": distance,
        "Weather": weather,
        "Traffic_Level": traffic,
        "Time_of_Day": time_of_day,
        "Vehicle_Type": vehicle,
        "Preparation_Time_min": preparation,
        "Courier_Experience_yrs": experience,
        "Weather_Traffic": weather_traffic
    }])

    prediction = model.predict(input_data)[0]

    st.success("Prediksi berhasil!")

    st.metric(
        "Estimasi Waktu Pengiriman",
        f"{prediction:.2f} menit"
    )

    st.info(
        "Hasil merupakan estimasi berdasarkan model Machine Learning."
    )

st.caption("Mini Project Machine Learning - Food Delivery Time Prediction")
