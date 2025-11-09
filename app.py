import streamlit as st
import joblib
import numpy as np
import pandas as pd
from geopy.geocoders import OpenCage
import matplotlib.pyplot as plt
import datetime
import time
from geopy.exc import GeocoderUnavailable, GeocoderServiceError

# Load the trained model
mag_model = joblib.load("random_forest_regressor.pkl")

# --- Set Page Config ---
st.set_page_config(page_title="Earthquake Magnitude Predictor", layout="wide")
st.markdown("<style>.stApp {background-color:#f0f8ff;}</style>", unsafe_allow_html=True)
st.markdown("""<style>.center-table {margin-left: auto; margin-right: auto; margin-top:0px; text-align: center;}
                .center-table th, .center-table td {text-align: center !important; padding:3px; }
                </style>""", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; margin-top:0px; padding-top:0px;'>Earthquake Magnitude Predictor</h2>",
            unsafe_allow_html=True)
st.write("<h4 style='text-align: center; margin-top:0px; padding-top:0px;'><i>Transforming Seismic Data into "
         "Life-Saving Insights</i></h4>", unsafe_allow_html=True)

months = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
}
month_names = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
}

# ==============================================
# ENTER DETAILS SECTION
# ==============================================
col1, col2, col3, col4, col5 = st.columns(5, vertical_alignment="bottom")
with col1:
    st.write("<h4 style='margin-top:0px; padding-top:0px;'>Enter Details</h4>", unsafe_allow_html=True)
with col2:
    start_date = datetime.date(2026, 1, 1)
    end_date = datetime.date(2026, 12, 31)
    inp_date = st.date_input("Choose date", min_value=start_date, max_value=end_date)

    inp_month = inp_date.month
    inp_year = inp_date.year
    inp_month_name = month_names[inp_month]
with col3:
    inp_latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=0.00, step=0.01, format="%0.1f")
with col4:
    inp_longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=0.00, step=0.01,
                                    format="%0.1f")
with col5:
    submit = st.button("Predict")

lat_grid = round(inp_latitude, 1)
lon_grid = round(inp_longitude, 1)
# ==============================================

# ==============================================
# PREDICTION SECTION
# ==============================================
col1, col2 = st.columns([2, 1])
with col1:
    st.write("")
    st.write("<h4 style='margin-top:0px; padding-top:0px;'>Prediction</h4>", unsafe_allow_html=True)

    if submit:
        input_data = np.array([[inp_month, lat_grid, lon_grid]])
        predicted_mag = np.round(mag_model.predict(input_data)[0], 2)

        # ==================================================
        # Computing Prediction Interval
        # ==================================================
        all_tree_predictions = np.array([tree.predict(input_data)[0] for tree in mag_model.estimators_])
        predicted_mean_mag = np.mean(all_tree_predictions)
        std_dev = np.std(all_tree_predictions)
        lower = predicted_mean_mag - 1.96 * std_dev
        upper = predicted_mean_mag + 1.96 * std_dev
        # ==================================================

        # ==================================================
        # Extracting location details, like city and country
        # (Now using OpenCage for reliability)
        # ==================================================
        API_KEY = "31713ef3db3e44c884c35a1caa51466b"  # 🔑 Replace this with your actual API key
        geolocator = OpenCage(api_key=API_KEY)

        location = None
        for attempt in range(3):
            try:
                location = geolocator.reverse((lat_grid, lon_grid), exactly_one=True, language='en', timeout=10)
                if location:
                    break
            except (GeocoderUnavailable, GeocoderServiceError, ConnectionError, TimeoutError):
                time.sleep(2)
            except Exception:
                break

        try:
            if location and 'components' in location.raw:
                address = location.raw['components']
                city = (
                    address.get('city')
                    or address.get('town')
                    or address.get('village')
                    or address.get('municipality')
                    or 'Unknown'
                )
                country = address.get('country', 'Unknown')
            else:
                city, country = 'Unknown', 'Unknown'
        except Exception:
            city, country = 'Unknown', 'Unknown'
        # ==================================================

        # ==================================================
        # Display Output
        # ==================================================
        col11, col12 = st.columns([1.5, 1])
        with col11:
            st.success(
                f"Earthquake of magnitude **{predicted_mag}** is predicted at **{city}**, **{country}** in "
                f"**{inp_month_name}, 2026**  \n95% Prediction Interval: [**{lower:.2f}, {upper:.2f}**]"
            )

            if predicted_mag <= 4.0:
                st.info(f"**Low Magnitude (Minor)** \n- Usually not felt. \n- Little to no damage.")
            elif 4.0 < predicted_mag <= 5.9:
                st.warning(f"**Moderate Magnitude** \n- Can cause minor to moderate damage, especially near the "
                           f"epicenter. \n- Buildings may shake, and some structures could develop cracks")
            else:
                st.error(
                    f"**High Magnitude (Major)** \n- Can cause significant to catastrophic damage, particularly in "
                    f"populated or poorly constructed areas. \n- Aftershocks and tsunamis may also occur.")

        with col12:
            usgs_df = pd.read_csv("USGS_processed_2.csv")
            usgs_df['Lat_grid'] = usgs_df['Latitude'].round(1)
            usgs_df['Lon_grid'] = usgs_df['Longitude'].round(1)
            usgs_df['Date'] = pd.to_datetime(usgs_df['Date'])

            filtered = usgs_df[
                (usgs_df['Lat_grid'] == lat_grid) &
                (usgs_df['Lon_grid'] == lon_grid)
            ].sort_values(by='Date', ascending=False)

            st.write(f"<h6 style='margin-top:0px; padding-top:0px; text-align:center; font-weight:normal;'>"
                     f"Past Earthquakes at<br><b style='font-size:16px; color:blue;'>Latitude: {lat_grid}, Longitude: {lon_grid}</b>"
                     f"<br> (Max. 5)</h4>",
                     unsafe_allow_html=True)

            if filtered.empty:
                st.write("<h6 style='margin-top:0px; padding-top:0px;'>No matching records found.</h4>",
                         unsafe_allow_html=True)
            else:
                df_last5 = filtered[['Date', 'Magnitude']].head(5)
                df_last5['Date'] = pd.to_datetime(df_last5['Date']).dt.strftime('%d-%b-%Y')
                html_table = df_last5.to_html(index=False, classes='center-table')
                st.markdown(html_table, unsafe_allow_html=True)

        # ===========================================================
        # Trend Forecast for next 12 months
        # ===========================================================
        start_date = pd.Timestamp(year=inp_year, month=inp_month, day=1)
        end_date = pd.Timestamp(year=2026, month=12, day=31)
        future_dates = pd.date_range(start=start_date, end=end_date, freq="M")
        forecast_features = [[d.month, lat_grid, lon_grid] for d in future_dates]

        all_preds = np.array([
            [tree.predict([f])[0] for tree in mag_model.estimators_]
            for f in forecast_features
        ])
        mean_preds = np.mean(all_preds, axis=1)
        std_preds = np.std(all_preds, axis=1)
        lower = mean_preds - 1.96 * std_preds
        upper = mean_preds + 1.96 * std_preds

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(future_dates, mean_preds, marker="o", label="Predicted Magnitude", color="blue")
        ax.fill_between(future_dates, lower, upper, color="lightblue", alpha=0.4, label="95% Prediction Interval")

        month_labels = [d.strftime("%b") for d in future_dates]
        ax.set_xticks(future_dates)
        ax.set_xticklabels(month_labels, rotation=0)

        ax.set_title(f"Earthquake Magnitude Forecast (Latitude: {inp_latitude}, Longitude: {inp_longitude})"
                     f" up to Dec 2026")
        ax.set_xlabel("Month")
        ax.set_ylabel("Predicted Magnitude")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        st.pyplot(fig)

with col2:
    if submit:
        location = pd.DataFrame({'lat': [inp_latitude], 'lon': [inp_longitude]})
        st.map(location, zoom=6)


