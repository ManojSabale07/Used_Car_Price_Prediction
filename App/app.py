import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ── Load model and column list ──────────────────────────────────────────────
model = joblib.load('model.pkl')
model_columns = joblib.load('model_columns.pkl')

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="centered")

st.title("🚗 Used Car Price Predictor")
st.write("Estimate the fair market value of a used car — powered by XGBoost (R² = 92.21%)")
st.divider()

# ── Input form ──────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    year = st.slider("Year of Manufacture", 2000, 2024, 2018)
    km_driven = st.number_input("KM Driven", min_value=0, max_value=500000,
                                 value=45000, step=1000)
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "LPG", "CNG"])
    seller_type = st.selectbox("Seller Type",
                                ["Individual", "Dealer", "Trustmark Dealer"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

with col2:
    mileage = st.number_input("Mileage (kmpl)", min_value=5.0,
                               max_value=50.0, value=18.0, step=0.5)
    engine = st.number_input("Engine (CC)", min_value=500,
                              max_value=5000, value=1200, step=100)
    max_power = st.number_input("Max Power (bhp)", min_value=30.0,
                                 max_value=600.0, value=85.0, step=5.0)
    seats = st.selectbox("Seats", [2, 4, 5, 6, 7, 8, 9, 10], index=2)
    owner = st.selectbox("Ownership",
                          ["First Owner", "Second Owner",
                           "Third Owner", "Fourth & Above Owner"])

brand = st.selectbox("Car Brand", [
    "Maruti", "Hyundai", "Honda", "Toyota", "Ford", "Tata",
    "Mahindra", "Volkswagen", "Skoda", "Renault", "Nissan",
    "Mercedes-Benz", "BMW", "Audi", "Kia", "MG", "Jeep",
    "Volvo", "Land", "Lexus", "Mitsubishi", "Chevrolet",
    "Fiat", "Force", "Isuzu", "Jaguar", "Opel",
    "Datsun", "Daewoo", "Ashok"
])

st.divider()

# ── Prediction ───────────────────────────────────────────────────────────────
if st.button("Predict Price", type="primary", use_container_width=True):

    # -- owner ordinal encoding (same as training) --
    owner_map = {
        'First Owner': 1,
        'Second Owner': 2,
        'Third Owner': 3,
        'Fourth & Above Owner': 4
    }

    # -- derived features (same as training) --
    age = 2025 - year
    log_km_driven = np.log1p(km_driven)

    # -- build base row with numeric features --
    input_dict = {
        'owner':         owner_map[owner],
        'mileage':       mileage,
        'engine':        engine,
        'max_power':     max_power,
        'seats':         seats,
        'log_km_driven': log_km_driven,
        'age':           age,
    }
    input_df = pd.DataFrame([input_dict])

    # -- add all dummy columns as 0 --
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # -- set seller_type dummy --
    if seller_type == "Individual":
        input_df['seller_type_Individual'] = 1
    elif seller_type == "Trustmark Dealer":
        input_df['seller_type_Trustmark Dealer'] = 1
    # "Dealer" is the drop_first reference → stays 0

    # -- set fuel dummy --
    if fuel == "Diesel":
        input_df['fuel_Diesel'] = 1
    elif fuel == "LPG":
        input_df['fuel_LPG'] = 1
    elif fuel == "Petrol":
        input_df['fuel_Petrol'] = 1
    # "CNG" is drop_first reference → stays 0

    # -- set transmission dummy --
    if transmission == "Manual":
        input_df['transmission_Manual'] = 1
    # "Automatic" is drop_first reference → stays 0

    # -- set brand dummy --
    brand_col = f'brand_{brand}'
    if brand_col in model_columns:
        input_df[brand_col] = 1
    # brands not in model_columns → stays 0 (treated as reference brand)

    # -- reorder columns to exactly match training --
    input_df = input_df[model_columns]

    # -- predict (model trained on log target, reverse with expm1) --
    log_pred = model.predict(input_df)[0]
    predicted_price = np.expm1(log_pred)

    # -- display result --
    st.success(f"### Estimated Selling Price: ₹{predicted_price:,.0f}")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Predicted Price", f"₹{predicted_price:,.0f}")
    col_b.metric("Car Age", f"{age} years")
    col_c.metric("KM Driven", f"{km_driven:,} km")

    st.caption(
        "Model: XGBoost | R² = 92.21% | Dataset: CarDekho (8,128 cars) | "
        "Predictions are estimates — actual prices may vary based on condition and location."
    )