import json
import pickle
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# --- Config ---
st.set_page_config(
    page_title="Salary Analytics Dashboard",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Paths & Data Loading ---
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "salary_model.pkl"
METADATA_PATH = ROOT / "model_metadata.json"

@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)

@st.cache_data
def load_metadata():
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)

model = load_model()
metadata = load_metadata()

# --- Constants ---
EXCHANGE_RATES = {
    "USD": {"rate": 1, "symbol": "$"},
    "EUR": {"rate": 0.92, "symbol": "€"},
    "GBP": {"rate": 0.79, "symbol": "£"},
    "INR": {"rate": 83.5, "symbol": "₹"},
    "CAD": {"rate": 1.37, "symbol": "C$"},
    "AUD": {"rate": 1.52, "symbol": "A$"}
}

def format_salary(amount, currency):
    rate_info = EXCHANGE_RATES[currency]
    converted = amount * rate_info["rate"]
    return f"{rate_info['symbol']}{converted:,.2f}"

# --- Sidebar (Dataset Snapshot) ---
st.sidebar.title("Dataset Snapshot")
st.sidebar.markdown("---")
st.sidebar.metric(label="Total records", value="23,435")
st.sidebar.metric(label="Unique countries", value="166")
st.sidebar.metric(label="Remote categories", value="3")
st.sidebar.metric(label="Average salary", value="$86,155")
st.sidebar.markdown("---")
st.sidebar.caption("Data sourced from global software engineering surveys.")

# --- Main Dashboard ---
st.markdown("<h4 style='color: #3b82f6;'>Salary analytics dashboard</h4>", unsafe_allow_html=True)
st.title("Salary trends from software engineering survey data")
st.write("A polished business dashboard with clean spacing, classic colors, and interactive visual summaries built on the existing dataset.")

st.markdown("---")

# --- Layout: Prediction & Charts ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Instant salary estimator")
    
    with st.form("prediction_form"):
        country = st.selectbox("Country", options=metadata.get("countries", []))
        remote_work = st.selectbox("Remote work style", options=metadata.get("remoteWorkOptions", []))
        years = st.number_input("Years of Experience", min_value=0.0, step=0.5, value=5.0)
        
        submitted = st.form_submit_button("Estimate salary", use_container_width=True)

    currency = st.selectbox("Display Currency", options=list(EXCHANGE_RATES.keys()), index=0)

    if submitted:
        if country and remote_work:
            features = pd.DataFrame([{
                "Country": country,
                "YearsCodePro": float(years),
                "RemoteWork": remote_work,
            }])
            
            try:
                predicted_value = model.predict(features)[0]
                formatted_salary = format_salary(predicted_value, currency)
                
                st.success(f"### Estimated yearly salary: {formatted_salary}")
                
                # Trajectory calculation
                trajectory_years = [1, 3, 5, 10, 15]
                trajectory_features = pd.DataFrame([{
                    "Country": country,
                    "YearsCodePro": float(y),
                    "RemoteWork": remote_work,
                } for y in trajectory_years])
                
                trajectory_preds = model.predict(trajectory_features)
                
                # Plotly Chart
                rate = EXCHANGE_RATES[currency]["rate"]
                symbol = EXCHANGE_RATES[currency]["symbol"]
                y_data = [p * rate for p in trajectory_preds]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=trajectory_years,
                    y=y_data,
                    mode='lines+markers',
                    line=dict(color='#3b82f6', width=3, shape='spline'),
                    marker=dict(color='#60a5fa', size=8),
                    fill='tozeroy',
                    fillcolor='rgba(59, 130, 246, 0.1)'
                ))
                
                fig.update_layout(
                    title="Career Trajectory Estimate",
                    xaxis_title="Years of Experience",
                    yaxis_title=f"Salary ({symbol})",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=40, r=20, t=40, b=40)
                )
                
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#334155')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#334155', zerolinecolor='#334155')
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")

with col2:
    st.subheader("Top 10 countries by average salary")
    
    # Hardcoded data from script.js
    countries = ['Gabon', 'Ethiopia', 'United States', 'Singapore', 'South Africa', 'Taiwan', 'Antigua & Barbuda', 'Andorra', 'Israel', 'Switzerland']
    salaries = [2000000, 930000, 160000, 135000, 135000, 130000, 130000, 125000, 125000, 125000]
    
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=countries,
        y=salaries,
        marker_color='#3b82f6'
    ))
    
    fig_bar.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Country",
        yaxis_title="Average Salary ($)",
        xaxis_tickangle=-45,
        margin=dict(l=40, r=20, t=40, b=80)
    )
    
    fig_bar.update_xaxes(showgrid=False)
    fig_bar.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#334155', zerolinecolor='#334155')
    
    st.plotly_chart(fig_bar, use_container_width=True)
