"""
RentShield Germany - Streamlit Frontend
Main application entry point.
"""

import streamlit as st
import requests
from typing import Optional
import os

# Configuration
API_URL = os.getenv("RENTSHIELD_API_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="RentShield Germany",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    /* Main theme */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Headers */
    h1 {
        background: linear-gradient(90deg, #00d4ff, #7b2cbf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* Cards */
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
    }
    
    /* Risk badges */
    .risk-low {
        background: linear-gradient(135deg, #00c853, #69f0ae);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #ffc107, #ffeb3b);
        color: black;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff5722, #ff9800);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
    }
    
    .risk-critical {
        background: linear-gradient(135deg, #d32f2f, #f44336);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0f23 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #7b2cbf, #00d4ff);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(123, 44, 191, 0.4);
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: white;
    }
    
    /* Success/Error boxes */
    .success-box {
        background: rgba(0, 200, 83, 0.1);
        border-left: 4px solid #00c853;
        padding: 16px;
        border-radius: 8px;
        margin: 16px 0;
    }
    
    .warning-box {
        background: rgba(255, 193, 7, 0.1);
        border-left: 4px solid #ffc107;
        padding: 16px;
        border-radius: 8px;
        margin: 16px 0;
    }
    
    .error-box {
        background: rgba(244, 67, 54, 0.1);
        border-left: 4px solid #f44336;
        padding: 16px;
        border-radius: 8px;
        margin: 16px 0;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health() -> bool:
    """Check if the FastAPI backend is available."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def get_risk_color(risk_level: str) -> str:
    """Get color for risk level."""
    colors = {
        "LOW": "#00c853",
        "MEDIUM": "#ffc107",
        "HIGH": "#ff5722",
        "CRITICAL": "#d32f2f"
    }
    return colors.get(risk_level, "#9e9e9e")


def get_risk_emoji(risk_level: str) -> str:
    """Get emoji for risk level."""
    emojis = {
        "LOW": "✅",
        "MEDIUM": "⚠️",
        "HIGH": "🚨",
        "CRITICAL": "❌"
    }
    return emojis.get(risk_level, "❓")


# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/home.png", width=80)
    st.title("RentShield")
    st.caption("🇩🇪 Germany Rental Protection")
    
    st.divider()
    
    # API Status
    api_status = check_api_health()
    if api_status:
        st.success("🟢 API Connected")
    else:
        st.error("🔴 API Offline")
        st.caption(f"URL: {API_URL}")
    
    st.divider()
    
    st.markdown("### 🔗 Quick Links")
    st.page_link("pages/1_Single_Checker.py", label="🔍 Single Checker", icon="🏠")
    st.page_link("pages/2_Bulk_CSV.py", label="📊 Bulk Analysis", icon="📁")
    st.page_link("pages/3_Market_Insights.py", label="📈 Market Insights", icon="📊")
    
    st.divider()
    
    st.markdown("### ℹ️ About")
    st.caption("""
    RentShield Germany helps you identify 
    overpriced and potentially fraudulent 
    rental listings using advanced ML.
    """)
    
    st.divider()
    st.caption("v1.0.0 | Made with ❤️")


# Main content
st.title("🏠 RentShield Germany")
st.subheader("Protect yourself from overpriced and scam rental listings")

st.markdown("""
---

### Welcome to RentShield! 👋

RentShield Germany is your AI-powered companion for navigating the German rental market safely. 
Our system analyzes rental listings to detect:

""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(123, 44, 191, 0.2), rgba(0, 212, 255, 0.2)); 
                padding: 24px; border-radius: 16px; text-align: center;">
        <h2>💰</h2>
        <h4>Overpricing</h4>
        <p style="color: #aaa;">Compare against fair market rates</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(123, 44, 191, 0.2), rgba(0, 212, 255, 0.2)); 
                padding: 24px; border-radius: 16px; text-align: center;">
        <h2>🚨</h2>
        <h4>Scam Detection</h4>
        <p style="color: #aaa;">AI-powered fraud analysis</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(123, 44, 191, 0.2), rgba(0, 212, 255, 0.2)); 
                padding: 24px; border-radius: 16px; text-align: center;">
        <h2>📊</h2>
        <h4>Market Insights</h4>
        <p style="color: #aaa;">Understand local pricing</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
---

### 🚀 Get Started

Choose a feature from the sidebar to begin analyzing listings:

1. **Single Checker** - Analyze individual listings with detailed reports
2. **Bulk Analysis** - Upload CSV files for batch processing  
3. **Market Insights** - Explore pricing trends and statistics

---

### 🔒 How It Works

Our ML models analyze multiple factors to assess risk:

| Component | Method | Weight |
|-----------|--------|--------|
| **Text Analysis** | NLP + TF-IDF Classification | 50% |
| **Price Anomalies** | Isolation Forest | 30% |
| **Red Flag Rules** | Keyword Detection | 20% |

The combined score determines the risk level, helping you make informed decisions.

---

### ⚠️ Risk Levels Explained

""")

risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)

with risk_col1:
    st.success("**LOW** ✅")
    st.caption("Listing appears legitimate")
    
with risk_col2:
    st.warning("**MEDIUM** ⚠️")
    st.caption("Some concerns, verify")

with risk_col3:
    st.error("**HIGH** 🚨")
    st.caption("Multiple red flags")
    
with risk_col4:
    st.error("**CRITICAL** ❌")
    st.caption("Avoid this listing")

st.markdown("---")

# Footer
st.caption("""
⚠️ **Disclaimer**: RentShield provides risk assessments based on ML models and should be used as one 
factor in your decision-making process. Always verify listings through official channels and never 
send money without viewing the property in person.
""")
