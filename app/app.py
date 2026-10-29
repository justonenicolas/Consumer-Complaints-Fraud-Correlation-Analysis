import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent   # raíz del proyecto
DATA_DIR = BASE_DIR / "data" / "processed_data"

# Page Setup
st.set_page_config(
    page_title="Fraud Anomaly and Bank Consumer Complaints Detector",
    page_icon="🛡️",
    layout="wide"
)

# Main Title
st.title("🛡️ Early Alert Monitoring System")
st.markdown("""
This app was designed to numerically measure the possible relation that may exist between **Customer Complaint Topics (processed through NLP)**
and **Unsupervised Anomaly Detection in bank transactions** to predict fraud transactions
""")

# Data Loading
@st.cache_data
def load_data():
    enriched_complaints = pd.read_csv(DATA_DIR / "fraud_topic_enriched_complaints.csv", sep=";")
    merged = pd.read_csv(DATA_DIR / "merged_transaction_complaints.csv", sep=";") # Export from Notebook 5
    anomalies = pd.read_csv(DATA_DIR / "anomaly_classification_enriched.csv", sep=";") # Export from Notebook 4
    return enriched_complaints, merged, anomalies

try:
    enriched_complaints, merged, anomalies = load_data()
    merged["date_received"] = pd.to_datetime(merged["date_received"])
except Exception as e:
    st.warning("Make sure processed data is stored in `data/processed_data/`.", e)
    st.stop()

# Layout con Pestañas
tab1, tab2, tab3 = st.tabs(["📈 Monitoring & Alerts", "🗣️ Complaint Analysis (NLP)", "🤖 ML Models Evaluation"])

# ---------------------------------------------------------
# TAB 1: Monitoring & Alerts
# ---------------------------------------------------------
with tab1:
    st.subheader("Synchronized Trending")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Complaints", f"{len(enriched_complaints):,}")
    col2.metric("Avg Daily Complaint Volume", f"{merged['fraud_complaint_volume'].mean():.1f}")
    col3.metric("Daily Transactional Anomalies", f"{merged['autoencoder_anomaly'].mean():.1f}")
    
    # Interactive Plotly Graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=merged["date_received"], 
        y=merged["fraud_complaint_volume"],
        name="Fraud Complaints (NLP)",
        line=dict(color="royalblue", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=merged["date_received"], 
        y=merged["autoencoder_anomaly"],
        name="Autoencoder Anomalies",
        line=dict(color="firebrick", width=2, dash="dash"),
        yaxis="y2"
    ))
    
    fig.update_layout(
        title="Cross Pattern Detection Time Series",
        xaxis_title="Date",
        yaxis_title="Complaint Volume (NLP)",
        yaxis2=dict(title="Transaction Anomalies", overlaying="y", side="right"),
        legend=dict(x=0, y=1.1, orientation="h")
    )
    st.plotly_chart(fig, width='stretch')

# ---------------------------------------------------------
# TAB 2: Complaint Analysis (NLP)
# ---------------------------------------------------------
with tab2:
    st.subheader("BERTopic Topic Exploration")
    topico_selected = st.selectbox("Choose a topic:", enriched_complaints["topic_name"].unique())
    
    df_filtered = enriched_complaints[enriched_complaints["topic_name"] == topico_selected]
    st.write(f"**Total complaints for this topico:** {len(df_filtered)}")
    
    st.markdown("### Narratives Sample:")
    for text in df_filtered["consumer_complaint_narrative"].dropna().sample(min(3, len(df_filtered))):
        st.info(f'"{text}"')

# ---------------------------------------------------------
# TAB 3: Model Evaluation 
# ---------------------------------------------------------
with tab3:
    st.subheader("Isolation Forest vs Autoencoder Performance Overview")
    st.write("PR-AUC Results for Non Supervised Transactional Fraud Detection:")
    
    col_a, col_b = st.columns(2)
    col_a.metric("Autoencoder PR-AUC", "0.5815", delta="+140% vs Baseline")
    col_b.metric("Isolation Forest PR-AUC", "0.2422", delta="Standard Baseline")

    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=anomalies["ae_anomaly_score"],
        y=anomalies["xgb_predicted_fraud_prob"],
        mode="markers",
        name="Autoencoder",
        marker=dict(color="orange", opacity=0.6, size=7)
    ))

    fig.add_trace(go.Scattergl(
        x=anomalies["if_anomaly_score"].abs(),
        y=anomalies["xgb_predicted_fraud_prob"],
        mode="markers",
        name="Isolation Forest",
        marker=dict(color="royalblue", opacity=0.6, size=7)
    ))

    fig.update_layout(
        title="Predicted Fraud Probability vs Anomaly Score for Identified Fraudulent Transactions (Class = 1)",
        xaxis_title="Calculated Anomaly Score",
        yaxis_title="Predicted Fraud Probability",
        legend=dict(x=0, y=1.1, orientation="h"),
        template="plotly_white"
    )

    st.plotly_chart(fig, width='stretch')