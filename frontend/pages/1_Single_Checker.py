"""
RentShield Germany - Single Listing Checker
Analyze individual rental listings with detailed reports.
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import os

# Configuration
API_URL = os.getenv("RENTSHIELD_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Single Checker | RentShield",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }
    
    .score-gauge {
        text-align: center;
        padding: 20px;
    }
    
    .reason-card {
        background: rgba(255, 87, 34, 0.1);
        border-left: 4px solid #ff5722;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


def create_gauge(value: float, title: str, color: str) -> go.Figure:
    """Create a gauge chart for score visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16, 'color': 'white'}},
        number={'suffix': '%', 'font': {'size': 24, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'white'},
            'bar': {'color': color},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 2,
            'bordercolor': 'rgba(255,255,255,0.2)',
            'steps': [
                {'range': [0, 25], 'color': 'rgba(0, 200, 83, 0.3)'},
                {'range': [25, 50], 'color': 'rgba(255, 193, 7, 0.3)'},
                {'range': [50, 75], 'color': 'rgba(255, 87, 34, 0.3)'},
                {'range': [75, 100], 'color': 'rgba(211, 47, 47, 0.3)'}
            ]
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=250,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def get_risk_badge(risk_level: str) -> str:
    """Get HTML badge for risk level."""
    styles = {
        "LOW": "background: linear-gradient(135deg, #00c853, #69f0ae); color: white;",
        "MEDIUM": "background: linear-gradient(135deg, #ffc107, #ffeb3b); color: black;",
        "HIGH": "background: linear-gradient(135deg, #ff5722, #ff9800); color: white;",
        "CRITICAL": "background: linear-gradient(135deg, #d32f2f, #f44336); color: white;"
    }
    
    emojis = {"LOW": "✅", "MEDIUM": "⚠️", "HIGH": "🚨", "CRITICAL": "❌"}
    
    style = styles.get(risk_level, "background: #9e9e9e;")
    emoji = emojis.get(risk_level, "❓")
    
    return f"""<span style="{style} padding: 8px 20px; border-radius: 20px; 
               font-weight: 600; font-size: 18px;">{emoji} {risk_level} RISK</span>"""


def analyze_listing(data: dict) -> dict:
    """Call FastAPI to analyze listing."""
    try:
        response = requests.post(
            f"{API_URL}/predict/full",
            json=data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None


# Page content
st.title("🔍 Single Listing Checker")
st.caption("Analyze individual rental listings for red flags and fair pricing")

st.divider()

# Input form
with st.form("listing_form"):
    st.subheader("📝 Listing Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        city = st.selectbox(
            "City",
            options=[
                "Berlin", "Munich", "Hamburg", "Frankfurt", "Cologne",
                "Stuttgart", "Düsseldorf", "Dresden", "Leipzig", "Hannover",
                "Nuremberg", "Essen", "Dortmund", "Other"
            ],
            index=0
        )
        
        postcode = st.text_input(
            "Postcode",
            value="10115",
            max_chars=5,
            help="German 5-digit postcode"
        )
        
        living_space = st.number_input(
            "Living Space (m²)",
            min_value=10.0,
            max_value=500.0,
            value=55.0,
            step=5.0
        )
        
        rooms = st.number_input(
            "Number of Rooms",
            min_value=1.0,
            max_value=10.0,
            value=2.0,
            step=0.5
        )
    
    with col2:
        rent = st.number_input(
            "Monthly Rent (€)",
            min_value=100.0,
            max_value=10000.0,
            value=950.0,
            step=50.0,
            help="Cold rent (Kaltmiete) in euros"
        )
        
        year_built = st.number_input(
            "Year Built (optional)",
            min_value=1800,
            max_value=2026,
            value=1990,
            step=1
        )
        
        floor = st.number_input(
            "Floor (optional)",
            min_value=0,
            max_value=50,
            value=3,
            step=1
        )
    
    st.subheader("📄 Listing Description")
    description = st.text_area(
        "Description Text",
        value="Beautiful apartment in city center. Available immediately.",
        height=100,
        help="Paste the listing description to analyze for red flags"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submitted = st.form_submit_button(
            "🔍 Analyze Listing",
            use_container_width=True,
            type="primary"
        )

st.divider()

# Results
if submitted:
    with st.spinner("Analyzing listing..."):
        data = {
            "city": city,
            "postcode": postcode,
            "living_space": living_space,
            "rooms": rooms,
            "rent": rent,
            "description": description,
            "year_built": year_built,
            "floor": floor
        }
        
        result = analyze_listing(data)
    
    if result:
        st.subheader("📊 Analysis Results")
        
        # Risk level badge
        risk_badge = get_risk_badge(result["risk_level"])
        st.markdown(f"<div style='text-align: center; margin: 20px 0;'>{risk_badge}</div>", 
                   unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Fair Rent Estimate",
                f"€{result['predicted_rent']:.0f}",
                help="Our ML model's estimate of fair market rent"
            )
        
        with col2:
            delta_color = "inverse" if result["overpricing_pct"] > 0 else "normal"
            st.metric(
                "Overpricing",
                f"{result['overpricing_pct']*100:+.1f}%",
                delta=f"vs €{result['predicted_rent']:.0f} fair",
                delta_color=delta_color
            )
        
        with col3:
            st.metric(
                "Scam Score",
                f"{result['scam_score']*100:.0f}%",
                help="Combined risk score from all analysis methods"
            )
        
        with col4:
            st.metric(
                "Price per m²",
                f"€{result['actual_rent_per_sqm']:.2f}",
                delta=f"Fair: €{result['predicted_rent_per_sqm']:.2f}"
            )
        
        st.divider()
        
        # Score breakdown
        st.subheader("📈 Score Breakdown")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = create_gauge(result["nlp_score"], "NLP Analysis", "#7b2cbf")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Text-based scam detection")
        
        with col2:
            fig = create_gauge(result["anomaly_score"], "Anomaly Score", "#00d4ff")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Price/size anomaly detection")
        
        with col3:
            fig = create_gauge(result["rules_score"], "Red Flags", "#ff5722")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Rule-based detection")
        
        # Detected issues
        if result["reasons"]:
            st.divider()
            st.subheader("⚠️ Detected Issues")
            
            for reason in result["reasons"]:
                st.markdown(f"""
                <div style="background: rgba(255, 87, 34, 0.1); 
                            border-left: 4px solid #ff5722;
                            padding: 12px 16px; border-radius: 8px; 
                            margin: 8px 0;">
                    🚨 {reason}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No major red flags detected!")
        
        # Recommendations
        st.divider()
        st.subheader("💡 Recommendations")
        
        if result["risk_level"] == "LOW":
            st.success("""
            **This listing appears legitimate.** 
            
            - The price is within market range
            - No suspicious patterns detected
            - Still verify the property in person before signing
            """)
        elif result["risk_level"] == "MEDIUM":
            st.warning("""
            **Proceed with caution.**
            
            - Request additional information from the landlord
            - Schedule an in-person viewing
            - Verify the landlord's identity
            - Never pay before seeing the apartment
            """)
        elif result["risk_level"] == "HIGH":
            st.error("""
            **Multiple concerns detected.**
            
            - Do NOT send any money without verification
            - Request verified identification from landlord
            - Consider using official platforms like ImmoScout24
            - Report suspicious listings to local authorities
            """)
        else:  # CRITICAL
            st.error("""
            **⚠️ AVOID THIS LISTING**
            
            This listing shows strong indicators of a scam:
            - Do NOT send any money or personal information
            - Do NOT respond to requests for deposits
            - Report this listing to the platform
            - Consider reporting to local consumer protection
            """)

else:
    # Placeholder when no analysis done yet
    st.info("👆 Fill in the listing details above and click 'Analyze Listing' to get started.")
    
    with st.expander("ℹ️ What information do I need?"):
        st.markdown("""
        **Required:**
        - City and postcode
        - Living space in square meters
        - Number of rooms
        - Monthly rent (cold rent)
        
        **Optional but helpful:**
        - Year the building was constructed
        - Floor number
        - Listing description text (important for scam detection!)
        
        **Pro Tip:** Copy and paste the full listing description for the most accurate scam detection.
        """)
