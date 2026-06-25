import os
import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

st.set_page_config(page_title="Global Horizon Analytics Dashboard", layout="wide")

# --- SIGNATURE HOOKS & TITLE PRESENTATION ---
st.title("🌐 Global Horizon Analytics Dashboard")
st.markdown("**System Architecture Pipeline:** Multi-Country Macro Evaluation Engine & 10-Year Risk Simulation Horizons")
st.caption("Developed by: **Ertiza Abbas**")

PROCESSED_DATA_PATH = r"D:\python\Python\geoecon_project\data\processed"

# Sidebar Developer Context Branding
st.sidebar.markdown("### 🛠️ System Properties")
st.sidebar.info("Developed by: **Ertiza Abbas**")
st.sidebar.markdown("---")

# Sidebar Interface Controllers
st.sidebar.subheader("🌍 Country Profiling")
target_country = st.sidebar.selectbox("Select Profile Country Matrix:", ["Singapore", "Argentina"])
country_lower = target_country.lower()

st.sidebar.subheader("🛡️ Geopolitical Stress-Test Matrix Modifiers")
apply_shock = st.sidebar.checkbox("Activate Geopolitical Chokepoint Shock Scenario")
shock_magnitude = st.sidebar.slider("Trade Disruption Scale (% Reduction):", 10, 50, 25, step=5) if apply_shock else 0

processed_file = os.path.join(PROCESSED_DATA_PATH, f"{country_lower}_processed_matrix.csv")

if os.path.exists(processed_file):
    df = pd.read_csv(processed_file).sort_values("year").reset_index(drop=True)
    
    # Feature Engineering
    df['gdp_growth_lag1'] = df['gdp_growth_annual_pct'].shift(1)
    df['gdp_growth_lag2'] = df['gdp_growth_annual_pct'].shift(2)
    df['health_score_lag1'] = df['economic_health_score'].shift(1)
    df_ml = df.dropna().reset_index(drop=True)
    
    ml_features = ['year', 'gdp_growth_lag1', 'gdp_growth_lag2', 'health_score_lag1', 'working_age_pop_pct', 'gov_debt_gdp_pct']
    X = df_ml[ml_features]
    y = df_ml['economic_health_score']
    
    # Train Background XGBoost Instance
    model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, objective='reg:squarederror', random_state=42)
    model.fit(X, y)
    
    # 10-Year Rolling Forecast
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
    
    # --- AUTOMATED 1,000-RUN MONTE CARLO STOCHASTIC BANDS ---
    hist_volatility = float(df_ml['economic_health_score'].std())
    num_simulations = 1000
    sim_tracks = np.zeros((num_simulations, len(future_years)))
    
    for sim in range(num_simulations):
        c_score = lag1_health
        for t in range(len(future_years)):
            shock = np.random.normal(0, hist_volatility * 0.15)
            c_score = (c_score * 0.95 + lag1_health * 0.05) + shock
            sim_tracks[sim, t] = max(0, min(100, c_score))
            
    optimistic_band = np.percentile(sim_tracks, 95, axis=0)
    pessimistic_band = np.percentile(sim_tracks, 5, axis=0)

    # --- DISPLAY GRAPH WITH INTEGRATED SHADED BANDS ---
    st.subheader(f"📈 Macroeconomic Forecast & Probabilistic Risk Profile: {target_country}")
    fig, ax = plt.subplots(figsize=(12, 5), dpi=100)
    sns.set_theme(style="whitegrid")
    
    ax.plot(df_ml['year'].values, df_ml['economic_health_score'].values, label="Historical Actual Data", color="#1f77b4", linewidth=2, marker='o')
    ax.plot(forecast_df['year'].values, forecast_df['predicted_score'].values, label="XGBoost Base Forecast", color="#ff7f0e", linestyle="--", linewidth=2.5, marker='s')
    
    # Shading the Risk Confidence Bands
    ax.fill_between(future_years, pessimistic_band, optimistic_band, color="#ff7f0e", alpha=0.15, label="Monte Carlo Risk Horizon Bounds (5th-95th Pct)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower left")
    st.pyplot(fig)

    st.subheader(f"📊 Historical Timeline Overview: {target_country}")
    st.dataframe(df[['year', 'gdp_growth_annual_pct', 'inflation_rate_pct', 'gov_debt_gdp_pct', 'economic_health_score']].tail(5))

    # --- AUTOMATED REPORT GENERATION ENGINE (PDF) ---
    def generate_pdf_report(country, current_score, final_pred, worst_case):
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        p.setFont("Helvetica-Bold", 18)
        p.drawString(50, 750, f"GLOBAL HORIZON ANALYTICS PROFILE REPORT: {country.upper()}")
        p.setFont("Helvetica-Oblique", 10)
        p.drawString(50, 735, "Lead Systems Architecture Design Engineer: Ertiza Abbas")
        p.setStrokeColorRGB(0.1, 0.4, 0.7)
        p.line(50, 725, 550, 725)
        
        p.setFont("Helvetica-Bold", 12)
        p.drawString(50, 680, "1. Executive Summary Parameters Valuation Matrix")
        p.setFont("Helvetica", 10)
        p.drawString(70, 660, f"• Last Evaluated Historical Economic Health Rating: {current_score:.2f} / 100")
        p.drawString(70, 640, f"• Projected 10-Year Target Macro Horizon Score (2036 Base): {final_pred:.2f} / 100")
        p.drawString(70, 620, f"• Probabilistic Monte Carlo 5th Percentile Risk Threshold: {worst_case:.2f} / 100")
        
        p.setFont("Helvetica-Bold", 12)
        p.drawString(50, 570, "2. Analytical Strategic Outlook Methodology")
        p.setFont("Helvetica", 10)
        text_block = "This report evaluates macro-structural resilience by modeling sequential growth configurations against demographic transitions, trade chokepoint exposures, and fiscal sovereign debt vulnerabilities using tuned machine learning arrays."
        p.drawString(70, 550, text_block[:90])
        p.drawString(70, 535, text_block[90:])
        
        p.setFont("Helvetica-Oblique", 8)
        p.drawString(50, 50, "Classification: Confidential Operational Strategy Analytics Platform Document Output. Principal Developer: Ertiza Abbas.")
        p.showPage()
        p.save()
        buffer.seek(0)
        return buffer

    # Download Interface Element Button
    st.sidebar.subheader("📥 Export Executive Deliverables")
    pdf_data = generate_pdf_report(target_country, df_ml['economic_health_score'].iloc[-1], forecast_df['predicted_score'].iloc[-1], pessimistic_band[-1])
    st.sidebar.download_button(
        label="Download Analytical Executive Summary (PDF)",
        data=pdf_data,
        file_name=f"{country_lower}_macro_horizon_report.pdf",
        mime="application/pdf"
    )
    
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: gray;'>Global Horizon Analytics Framework | Developed by Ertiza Abbas</p>", unsafe_allow_html=True)
else:
    st.error("[-] Data file path mappings incorrect. Ensure data processing matrices exist on drive storage.")
