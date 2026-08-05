import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

st.set_page_config(
    page_title="IRCTC Flight Price Predictor",
    page_icon="✈️",
    layout="centered"
)

@st.cache_resource 
def load_models():
    try:
        # UPDATED to load q30 (the new file your local computer generated)
        models_q10 = joblib.load('route_models_q10.pkl')
        models_q30 = joblib.load('route_models_q30.pkl')
        return models_q10, models_q30
    except FileNotFoundError:
        return None, None

models_q10, models_q30 = load_models()

# --- SMART AIRPORT DICTIONARIES ---
home_airports = {
    'Kochi (COK)': 'COK',
    'Trivandrum (TRV)': 'TRV',
    'Calicut (CCJ)': 'CCJ',
    'Kannur (CNN)': 'CNN',
    'Coimbatore (CJB)': 'CJB'
}

tour_destinations = {
    'Delhi (DEL)': 'DEL', 'Bangalore (BLR)': 'BLR', 'Hyderabad (HYD)': 'HYD',
    'Chennai (MAA)': 'MAA', 'Patna (PAT)': 'PAT', 'Ayodhya (AYJ)': 'AYJ',
    'Guwahati (GAU)': 'GAU', 'Leh (IXL)': 'IXL', 'Srinagar (SXR)': 'SXR',
    'Dehradun (DED)': 'DED', 'Kuala Lumpur (KUL)': 'KUL', 'Bali (DPS)': 'DPS',
    'Seoul (ICN)': 'ICN', 'Tokyo (TYO)': 'TYO', 'Singapore (SIN)': 'SIN',
    'Colombo (CMB)': 'CMB', 'Gaya (GAY)': 'GAY', 'Lucknow (LKO)': 'LKO',
    'Gorakhpur (GOP)': 'GOP', 'Darbhanga (DBR)': 'DBR'
}

all_airports = {**home_airports, **tour_destinations}

st.title("✈️ Airfare Predictor")
st.markdown("Select your travel route and date to get an data powered baseline budget constraint.")
st.divider()

if models_q10 is None:
    st.error("❌ Model files not found! Please ensure 'route_models_q10.pkl' and 'route_models_q30.pkl' are in the same folder as this app.")
else:
    # --- DYNAMIC DROPDOWN LOGIC ---
    col1, col2 = st.columns(2)
    
    with col1:
        origin_ui = st.selectbox("🛫 Origin Airport", options=list(all_airports.keys()), index=0)
    
    with col2:
        if origin_ui in home_airports:
            dest_options = list(tour_destinations.keys())
        else:
            dest_options = list(home_airports.keys())
            
        dest_ui = st.selectbox("🛬 Destination Airport", options=dest_options)

    travel_date = st.date_input("📅 Select Travel Date")
    
    if st.button("🔮 Predict Baseline Fare", type="primary", use_container_width=True):
        
        origin_code = all_airports[origin_ui]
        dest_code = all_airports[dest_ui]
        route_code = f"{origin_code}_{dest_code}"
        
        current_date = datetime.now().date()
        days_ahead = (travel_date - current_date).days
        
        if origin_code == dest_code:
            st.warning("⚠️ Origin and Destination cannot be the same!")
        elif days_ahead < 0:
            st.error("❌ Travel date must be in the future!")
        elif route_code not in models_q10:
            st.error(f"❌ No historical budget data available to predict the route: {route_code}.")
        else:
            month = travel_date.month
            day_of_week = travel_date.weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
            
            input_dict = {
                'Days_To_Departure': [days_ahead],
                'Is_Weekend': [is_weekend],
                'Month': [month],
                'Day_Of_Week': [day_of_week] 
            }
            
            input_df = pd.DataFrame(input_dict)
            
            specific_model_q10 = models_q10[route_code]
            specific_model_q30 = models_q30[route_code]
            
            pred_floor = specific_model_q10.predict(input_df)[0]
            pred_baseline = specific_model_q30.predict(input_df)[0]
            
            if pred_baseline <= pred_floor:
                pred_baseline = pred_floor * 1.05
                
            st.success(f"✅ Prediction generated successfully for {route_code} ({days_ahead} Days Ahead)")
            
            st.markdown("### 📊 Market Cost Projections")
            
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.info("⚡ Absolute Lowest Expected Fare (Floor)")
                st.metric(label="Target Price", value=f"₹{round(pred_floor, -2):,.0f} – ₹{round(pred_floor * 1.05, -2):,.0f}")
                
            with res_col2:
                st.info("🟢 Realistic Budget Baseline Fare (Safe)")
                st.metric(label="Budget Constraint", value=f"₹{round(pred_baseline, -2):,.0f} – ₹{round(pred_baseline * 1.05, -2):,.0f}")
                
            st.caption("Note: Predictions are based on historical budget-carrier algorithms. Real-time dynamic pricing may cause slight variances.")
