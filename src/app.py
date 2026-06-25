import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# Set up page configurations
st.set_page_config(page_title="Global Horizon Analytics Dashboard", layout="wide")
st.title("🌐 Global Horizon Analytics Dashboard")
st.markdown("Interactive Macroeconomic Health Evaluation Engine & 10-Year Risk Simulation Horizons")

PROCESSED_DATA_PATH = r"D:\python\Python\geoecon_project\data\processed"

# Side-bar country navigation selectors
target_country = st.sidebar.selectbox("Select Profile Country Matrix:", ["Singapore", "Argentina"])
country_lower = target_country.lower()

processed_file = os.path.join(PROCESSED_DATA_PATH, f"{country_lower}_processed_matrix.csv")

if os.path.exists(processed_file):
    # Load processed asset matrix
    df = pd.read_csv(processed_file).sort_values("year").reset_index(drop=True)
    
    st.subheader(f"📊 Historical Timeline Overview: {target_country}")
    st.dataframe(df[['year', 'gdp_growth_annual_pct', 'inflation_rate_pct', 'gov_debt_gdp_pct', 'economic_health_score']].tail(5))
    
    # 1. Geopolitical Stress-Test Controller Interface Elements
    st.sidebar.subheader("🛡️ Geopolitical Stress-Test Matrix Modifiers")
    apply_shock = st.sidebar.checkbox("Activate Geopolitical Chokepoint Shock Scenario")
    shock_magnitude = st.sidebar.slider("Trade Disruption Scale (% Reduction):", 10, 50, 25, step=5) if apply_shock else 0

    # Feature engineering sequence
    df['gdp_growth_lag1'] = df['gdp_growth_annual_pct'].shift(1)
    df['gdp_growth_lag2'] = df['gdp_growth_annual_pct'].shift(2)
    df['health_score_lag1'] = df['economic_health_score'].shift(1)
    df_ml = df.dropna().reset_index(drop=True)
    
    ml_features = ['year', 'gdp_growth_lag1', 'gdp_growth_lag2', 'health_score_lag1', 'working_age_pop_pct', 'gov_debt_gdp_pct']
    X = df_ml[ml_features]
    y = df_ml['economic_health_score']
    
    # Train background model instant instance
    model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, objective='reg:squarederror', random_state=42)
    model.fit(X, y)
    
    # Run Predictions
    future_years = np.arange(2027, 2037)
    future_preds = []
    
    last_row = df_ml.iloc[-1]
    lag1_gdp = last_row['gdp_growth_annual_pct']
    lag2_gdp = last_row['gdp_growth_lag1']
    lag1_health = last_row['economic_health_score']
    working_pop = last_row['working_age_pop_pct']
    debt_gdp = last_row['gov_debt_gdp_pct']
    
    for yr in future_years:
        gdp_input = lag1_gdp * (1 - (shock_magnitude / 100)) if apply_shock and yr >= 2028 else lag1_gdp
        
        input_data = pd.DataFrame([{
            'year': yr, 'gdp_growth_lag1': gdp_input, 'gdp_growth_lag2': lag2_gdp,
            'health_score_lag1': lag1_health, 'working_age_pop_pct': working_pop, 'gov_debt_gdp_pct': debt_gdp
        }])
        
        pred_score = float(model.predict(input_data)[0])
        future_preds.append({"year": yr, "predicted_score": pred_score})
        
        lag2_gdp = gdp_input
        lag1_gdp = gdp_input * 0.95
        lag1_health = pred_score
        working_pop *= 0.995 if country_lower == "singapore" else 0.998
        
    forecast_df = pd.DataFrame(future_preds)
    
    # --- RENDER WEB GRAPHICAL PLOTS ---
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.set_theme(style="whitegrid")
    ax.plot(df_ml['year'].values, df_ml['economic_health_score'].values, label="Historical Actual Data Baseline", color="#1f77b4", linewidth=2, marker='o')
    ax.plot(forecast_df['year'].values, forecast_df['predicted_score'].values, label="XGBoost Dynamic Horizon Forecast", color="#ff7f0e", linestyle="--", linewidth=2.5, marker='s')
    ax.set_title(f"{target_country} Economic Health Score Projection Horizons", fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(loc="lower left")
    st.pyplot(fig)
    
else:
    st.error(f"[-] Data matrix file asset missing for choice: {target_country}. Check data storage directory paths.")
