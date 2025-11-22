import streamlit as st
import joblib
import numpy as np
import pandas as pd
from geopy.geocoders import OpenCage
import plotly.graph_objects as go
import matplotlib.pyplot as plt
#from datetime
import datetime

mag_model = joblib.load("quake_mag_model_10MB.pkl")

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
    # Define the range for 2026
    start_date = datetime.date(2026, 1, 1)
    end_date = datetime.date(2026, 12, 31)
    inp_date = st.date_input("Choose date", min_value=start_date, max_value=end_date)

    inp_day = inp_date.day
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

usgs_df = pd.read_csv("USGS_processed_2.csv")
col1, col2 = st.columns([2, 1])
with col1:
    st.write("")
    st.write("<h4 style='margin-top:0px; padding-top:0px;'>Prediction</h4>", unsafe_allow_html=True)

    if submit:
        input_data = np.array([[inp_month, lat_grid, lon_grid]])
        predicted_mag = np.round(mag_model.predict(input_data)[0], 2)

      
        all_tree_predictions = np.array([tree.predict(input_data)[0] for tree in mag_model.estimators_])
      
        predicted_mean_mag = np.mean(all_tree_predictions)
        std_dev = np.std(all_tree_predictions)
        
        lower = predicted_mean_mag - 1.96 * std_dev
        upper = predicted_mean_mag + 1.96 * std_dev
        
        API_KEY = "31713ef3db3e44c884c35a1caa51466b"  
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
        col11, col12 = st.columns([1.5, 1])
        with col11:
            # ====================
            # Predicted Magnitude
            # ====================
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
            # ===========================================================
            # Show past earthquakes for given location (max 5)
            # ===========================================================
            # Add grid columns and datetime
            usgs_df['Lat_grid'] = usgs_df['Latitude'].round(1)
            usgs_df['Lon_grid'] = usgs_df['Longitude'].round(1)
            usgs_df['Date'] = pd.to_datetime(usgs_df['Date'])

            # Filter for matching grid (any year)
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
                df_last5 = filtered[['Date', 'Magnitude']]
                # Format the Date column
                df_last5['Date'] = pd.to_datetime(df_last5['Date']).dt.strftime('%d-%b-%Y')

                # Convert to HTML with center alignment
                html_table = df_last5.to_html(index=False, classes='center-table')
                st.markdown(html_table, unsafe_allow_html=True)
            # ===========================================================

        tab1, tab2 = st.tabs(["Temporal Pattern of Earthquake Magnitudes",
                              "Earthquake Magnitude Trend Forecast up to December 2026"])
        with tab1:
            # ===========================================================
            # Temporal Pattern of Earthquake Magnitudes
            # ===========================================================
            st.write("<h4 style='margin-top:0px; padding-top:0px;'>Temporal Pattern of Earthquake Magnitudes</h4>",
                     unsafe_allow_html=True)

            # --- Define your time window ---
            cutoff_date = inp_date - pd.DateOffset(years=5)

            # --- Filter data within date range AND within 100 km ---
            # 1° of latitude ≈ 111 km; 1° of longitude ≈ 111 km * cos(latitude)
            deg_radius = 100 / 111  # ~0.45° for 50 km

            filtered_temp = usgs_df[
                (usgs_df['Date'] >= cutoff_date) &
                (usgs_df['Latitude'].between(lat_grid - deg_radius, lat_grid + deg_radius)) &
                (usgs_df['Longitude'].between(lon_grid - deg_radius, lon_grid + deg_radius))
                ]

            if filtered_temp.empty:
                st.warning("No historical earthquake data found for this region.")
            else:
                # --- Aggregate by month ---
                monthly = (
                    filtered_temp.groupby('Month')['Magnitude'].agg(['mean', 'std', 'count']).reset_index().sort_values('Month')
                )

                # Compute 95% confidence interval (mean ± 1.96 * std / sqrt(n))
                monthly['ci'] = 1.96 * monthly['std'] / np.sqrt(monthly['count'])
                monthly['Month_Name'] = monthly['Month'].map(month_names)

                st.write(f"**Summary | **Data points:** {len(filtered_temp)} "
                         f"| **Average magnitude:** {filtered_temp['Magnitude'].mean():.2f}"
                         f"| **Max magnitude:** {filtered_temp['Magnitude'].max():.2f}"
                         f"| **Min magnitude:** {filtered_temp['Magnitude'].min():.2f}")

                # --- Interactive plot with Plotly ---
                fig = go.Figure()

                # Main line (Mean magnitude line)
                fig.add_trace(go.Scatter(
                    x=monthly['Month_Name'], y=monthly['mean'],
                    mode='lines+markers', name='Mean Magnitude', line=dict(color='royalblue', width=2)
                ))

                fig.update_layout(
                    title=f"Average Earthquake Magnitude Trend (100 km radius)<br><sup>Lat: {inp_latitude}, Lon: {inp_longitude}</sup>",
                    xaxis_title="Month", yaxis_title="Average Magnitude",
                    template="plotly_white", height=400, margin=dict(l=20, r=20, t=60, b=40)
                )

                fig.update_yaxes(tickformat=".2f")
                st.plotly_chart(fig, use_container_width=True)
        with tab2:
            # ===========================================================
            # Trend Forecast up to December 2026
            # ===========================================================
            st.write("<h4 style='margin-top:0px; padding-top:0px;'>Earthquake Magnitude Trend "
                     "Forecast up to December 2026</h4>", unsafe_allow_html=True)
            # st.write(f"**Location Coordinates | **Latitude:** {inp_latitude} "
            #         f"| **Longitude:** {inp_longitude}")

            # Generate forecast up to Dec 2026
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

            # --- Plotly Interactive Plot ---
            fig = go.Figure()

            # Mean prediction line
            fig.add_trace(go.Scatter(x=future_dates, y=mean_preds, mode='lines+markers',
                name='Forecasted Magnitude', line=dict(color='blue'), marker=dict(size=6)
            ))

            # Confidence interval (shaded)
            fig.add_trace(go.Scatter(
                x=list(future_dates) + list(future_dates[::-1]),
                y=list(upper) + list(lower[::-1]),
                fill='toself', fillcolor='rgba(173, 216, 230, 0.4)',
                line=dict(color='rgba(255,255,255,0)'), hoverinfo='skip', name='95% CI'
            ))

            fig.update_layout(
                title=f"Earthquake Magnitude Forecast up to Dec 2026<br><sup>Lat: {inp_latitude}, Lon: {inp_longitude}</sup>",
                xaxis_title="Month", yaxis_title="Forecasted Magnitude",
                template="plotly_white", height=400, margin=dict(l=20, r=20, t=60, b=40)
            )

            # Format x-axis as month abbreviations
            fig.update_xaxes(tickmode='array', tickvals=future_dates,
                             ticktext=[d.strftime("%b") for d in future_dates])

            fig.update_yaxes(tickformat=".2f")
            #fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
with col2:
    if submit:
        location = pd.DataFrame({'lat': [inp_latitude], 'lon': [inp_longitude]})
        st.map(location, zoom=6)


