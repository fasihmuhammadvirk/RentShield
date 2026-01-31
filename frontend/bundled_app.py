"""
RentShield Germany - Bundled App (Serverless)
Runs both Frontend and Backend logic in a single Streamlit app.
Ideal for deployment on Streamlit Community Cloud.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to python path to allow imports from backend
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from backend.app.core.pricing import PricingPredictor
from backend.app.core.scam import ScamDetector

# Page configuration
st.set_page_config(
    page_title="RentShield Germany (Bundled)",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Same as main app)
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); }
    h1 { background: linear-gradient(90deg, #00d4ff, #7b2cbf); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; }
    .stMetric { background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 16px; }
    .risk-badge { padding: 8px 16px; border-radius: 20px; font-weight: 600; color: white; display: inline-block; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# CORE LOGIC LOADING (Replaces FastAPI)
# ------------------------------------------------------------------

@st.cache_resource
def load_models():
    """Load ML models once and cache them in memory."""
    import os
    import traceback
    
    # Debug: Show current working directory and file structure
    artifacts_path = root_path / "backend" / "app" / "artifacts"
    
    st.sidebar.write("**Debug Info:**")
    st.sidebar.write(f"Root path: `{root_path}`")
    st.sidebar.write(f"Artifacts path: `{artifacts_path}`")
    st.sidebar.write(f"Artifacts exists: {artifacts_path.exists()}")
    
    if artifacts_path.exists():
        st.sidebar.write(f"Artifacts files: {list(artifacts_path.glob('*'))}")
    
    error_messages = []
    
    try:
        # Initialize pricing predictor
        st.sidebar.write("Loading pricing model...")
        pricer = PricingPredictor(artifacts_path)
        pricer_loaded = pricer.load_model()
        
        if pricer_loaded:
            st.sidebar.write("✅ Pricing model loaded")
        else:
            st.sidebar.write("❌ Pricing model failed")
            error_messages.append("Pricing model failed to load")
        
    except Exception as e:
        pricer = None
        pricer_loaded = False
        error_msg = f"Pricing error: {str(e)}\n{traceback.format_exc()}"
        st.sidebar.error(error_msg)
        error_messages.append(error_msg)
    
    try:
        # Initialize scam detector
        st.sidebar.write("Loading scam detector...")
        detector = ScamDetector(artifacts_path)
        detector_loaded = detector.load_models()
        
        if detector_loaded:
            st.sidebar.write("✅ Scam detector loaded")
        else:
            st.sidebar.write("❌ Scam detector failed")
            error_messages.append("Scam detector failed to load")
            
    except Exception as e:
        detector = None
        detector_loaded = False
        error_msg = f"Scam detector error: {str(e)}\n{traceback.format_exc()}"
        st.sidebar.error(error_msg)
        error_messages.append(error_msg)
    
    models_ready = pricer_loaded and detector_loaded
    
    if models_ready:
        st.sidebar.success("✅ All models loaded successfully!")
    else:
        st.sidebar.error("⚠️ Some models failed to load")
        if error_messages:
            st.sidebar.write("**Error Details:**")
            for msg in error_messages:
                st.sidebar.code(msg)
    
    return pricer, detector, models_ready

pricing_model, scam_model, models_ready = load_models()

# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------

def analyze_listing(data: dict):
    """Directly calls backend classes instead of API endpoint."""
    if not models_ready:
        st.error("ML Models failed to load!")
        return None

    # 1. Price Analysis
    price_result, price_reasons = pricing_model.get_price_analysis(
        city=data.get('city'),
        postcode=data.get('postcode'),
        living_space=data.get('living_space'),
        rooms=data.get('rooms'),
        rent=data.get('rent'),
        year_built=data.get('year_built'),
        floor=data.get('floor')
    )

    # 2. Scam Analysis
    scam_result = scam_model.analyze(
        description=data.get('description', ""),
        rent=data.get('rent'),
        living_space=data.get('living_space'),
        predicted_rent=price_result["predicted_rent"]
    )

    # 3. Combine Results
    all_reasons = price_reasons + scam_result["reasons"]
    unique_reasons = list(dict.fromkeys(all_reasons))[:6]

    return {
        **price_result,
        **scam_result,
        "reasons": unique_reasons
    }

def get_risk_badge_html(level):
    colors = {
        "LOW": "linear-gradient(135deg, #00c853, #69f0ae)",
        "MEDIUM": "linear-gradient(135deg, #ffc107, #ffeb3b); color: black",
        "HIGH": "linear-gradient(135deg, #ff5722, #ff9800)",
        "CRITICAL": "linear-gradient(135deg, #d32f2f, #f44336)"
    }
    style = f"background: {colors.get(level, '#999')};"
    return f'<div class="risk-badge" style="{style}">{level} RISK</div>'

# ------------------------------------------------------------------
# UI LAYOUT
# ------------------------------------------------------------------

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/home.png", width=80)
    st.title("RentShield")
    st.caption("🇩🇪 Bundled Version")
    st.success("🟢 Running Locally")
    st.info("This version runs without a separate backend server.")

# Main Interface
st.title("RentShield Single Checker")

with st.form("check_form"):
    col1, col2 = st.columns(2)
    with col1:
        city = st.selectbox("City", ["Berlin", "Munich", "Hamburg", "Frankfurt", "Cologne", "Leipzig", "Other"])
        postcode = st.text_input("Postcode", "10115")
        size = st.number_input("Size (m²)", 10, 500, 55)
        rooms = st.number_input("Rooms", 1.0, 10.0, 2.0)
    with col2:
        rent = st.number_input("Cold Rent (€)", 100, 10000, 950)
        desc = st.text_area("Description", "Beautiful apartment...", height=150)
    
    submit = st.form_submit_button("Analyze Listing", type="primary")

if submit:
    with st.spinner("Analyzing..."):
        result = analyze_listing({
            "city": city, "postcode": postcode, 
            "living_space": size, "rooms": rooms, 
            "rent": rent, "description": desc,
            "year_built": 2000, "floor": 2
        })
    
    if result:
        st.markdown("---")
        
        # Header Stats
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.markdown(get_risk_badge_html(result['risk_level']), unsafe_allow_html=True)
        with c2:
            st.metric("Scam Score", f"{result['scam_score']*100:.0f}%")
        with c3:
            st.metric("Fair Rent Estimate", f"€{result['predicted_rent']:.0f}", 
                     delta=f"{result['overpricing_pct']*100:+.1f}% vs actual", delta_color="inverse")
            
        # Analysis
        st.subheader("Analysis Breakdown")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Price Analysis**")
            st.progress(max(0, min(1, result['overpricing_pct'] + 0.5)))
            st.caption(f"Market Price: €{result['predicted_rent_per_sqm']:.2f}/m²")
            
        with col2:
            st.write("**Text Analysis (NLP)**")
            st.progress(result['nlp_score'])
            st.caption(f"Suspicious Language: {result['nlp_score']*100:.0f}% chance")

        if result['reasons']:
            st.error("🚩 **Red Flags Detected:**")
            for r in result['reasons']:
                st.write(f"- {r}")
        else:
            st.success("✅ No obvious red flags detected.")

st.markdown("---")
st.caption("RentShield Germany | Bundled Deployment Mode")
