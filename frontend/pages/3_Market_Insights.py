"""
RentShield Germany - Market Insights
Explore rental market statistics and trends.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os

st.set_page_config(
    page_title="Market Insights | RentShield",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .insight-card {
        background: linear-gradient(135deg, rgba(0,212,255,0.1), rgba(123,44,191,0.1));
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
    }
</style>
""", unsafe_allow_html=True)


# Market data (based on 2024 German rental market)
MARKET_DATA = {
    "Munich": {"avg_rent": 21.50, "tier": 1, "population": 1500000, "trend": 0.05},
    "Frankfurt": {"avg_rent": 18.00, "tier": 1, "population": 750000, "trend": 0.04},
    "Stuttgart": {"avg_rent": 16.50, "tier": 1, "population": 630000, "trend": 0.03},
    "Berlin": {"avg_rent": 14.50, "tier": 2, "population": 3600000, "trend": 0.06},
    "Hamburg": {"avg_rent": 14.00, "tier": 2, "population": 1900000, "trend": 0.04},
    "Düsseldorf": {"avg_rent": 13.50, "tier": 2, "population": 620000, "trend": 0.03},
    "Cologne": {"avg_rent": 13.00, "tier": 2, "population": 1100000, "trend": 0.04},
    "Hannover": {"avg_rent": 10.50, "tier": 3, "population": 535000, "trend": 0.02},
    "Dresden": {"avg_rent": 9.00, "tier": 3, "population": 560000, "trend": 0.03},
    "Leipzig": {"avg_rent": 8.50, "tier": 3, "population": 600000, "trend": 0.05},
    "Nuremberg": {"avg_rent": 12.00, "tier": 3, "population": 520000, "trend": 0.02},
    "Essen": {"avg_rent": 8.00, "tier": 4, "population": 580000, "trend": 0.01},
    "Dortmund": {"avg_rent": 7.50, "tier": 4, "population": 600000, "trend": 0.01},
    "Duisburg": {"avg_rent": 6.50, "tier": 4, "population": 500000, "trend": 0.00},
}


# Page content
st.title("📈 Market Insights")
st.caption("Explore German rental market statistics and trends")

st.divider()

# Overview metrics
st.subheader("🌍 Market Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_rent = sum(d['avg_rent'] for d in MARKET_DATA.values()) / len(MARKET_DATA)
    st.metric(
        "Avg. Rent (Germany)",
        f"€{avg_rent:.2f}/m²",
        delta="+3.5% YoY"
    )

with col2:
    st.metric(
        "Most Expensive",
        "Munich",
        delta="€21.50/m²"
    )

with col3:
    st.metric(
        "Most Affordable",
        "Duisburg",
        delta="€6.50/m²"
    )

with col4:
    total_pop = sum(d['population'] for d in MARKET_DATA.values())
    st.metric(
        "Cities Covered",
        len(MARKET_DATA),
        delta=f"{total_pop/1000000:.1f}M people"
    )

st.divider()

# Price comparison chart
st.subheader("💰 Rent Prices by City")

df_cities = pd.DataFrame([
    {"City": city, **data}
    for city, data in MARKET_DATA.items()
]).sort_values("avg_rent", ascending=True)

# Create gradient colors based on tier
tier_colors = {1: "#d32f2f", 2: "#ff5722", 3: "#ffc107", 4: "#4caf50"}
df_cities['color'] = df_cities['tier'].map(tier_colors)

fig = px.bar(
    df_cities,
    x="avg_rent",
    y="City",
    orientation='h',
    color="tier",
    color_continuous_scale=["#4caf50", "#ffc107", "#ff5722", "#d32f2f"],
    labels={"avg_rent": "Average Rent (€/m²)", "tier": "Price Tier"}
)

fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font={'color': 'white'},
    height=500,
    showlegend=False
)

fig.update_traces(
    marker_line_width=0,
    hovertemplate="<b>%{y}</b><br>€%{x:.2f}/m²<extra></extra>"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# City tiers explanation
st.subheader("🏙️ City Tiers")

tier_col1, tier_col2 = st.columns(2)

with tier_col1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(211,47,47,0.2), rgba(211,47,47,0.05));
                padding: 20px; border-radius: 12px; margin: 10px 0;">
        <h4 style="color: #d32f2f;">🔴 Tier 1 - Premium Cities</h4>
        <p><strong>Munich, Frankfurt, Stuttgart</strong></p>
        <p>Highest rental prices, major economic centers, limited housing supply.</p>
        <p style="font-size: 24px; font-weight: bold;">€16-22/m²</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(255,193,7,0.2), rgba(255,193,7,0.05));
                padding: 20px; border-radius: 12px; margin: 10px 0;">
        <h4 style="color: #ffc107;">🟡 Tier 3 - Growing Cities</h4>
        <p><strong>Hannover, Dresden, Leipzig, Nuremberg</strong></p>
        <p>Developing markets with increasing demand and moderate prices.</p>
        <p style="font-size: 24px; font-weight: bold;">€8-12/m²</p>
    </div>
    """, unsafe_allow_html=True)

with tier_col2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(255,87,34,0.2), rgba(255,87,34,0.05));
                padding: 20px; border-radius: 12px; margin: 10px 0;">
        <h4 style="color: #ff5722;">🟠 Tier 2 - Major Cities</h4>
        <p><strong>Berlin, Hamburg, Düsseldorf, Cologne</strong></p>
        <p>Large metropolitan areas with strong rental demand.</p>
        <p style="font-size: 24px; font-weight: bold;">€13-15/m²</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(76,175,80,0.2), rgba(76,175,80,0.05));
                padding: 20px; border-radius: 12px; margin: 10px 0;">
        <h4 style="color: #4caf50;">🟢 Tier 4 - Affordable Cities</h4>
        <p><strong>Essen, Dortmund, Duisburg</strong></p>
        <p>Most affordable options, stable markets with good availability.</p>
        <p style="font-size: 24px; font-weight: bold;">€6-8/m²</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Trend analysis
st.subheader("📊 Price Trends (YoY)")

fig_trend = go.Figure()

# Sort cities by trend
df_trend = df_cities.sort_values("trend", ascending=False)

fig_trend.add_trace(go.Bar(
    x=df_trend["City"],
    y=df_trend["trend"] * 100,
    marker_color=[tier_colors[t] for t in df_trend["tier"]],
    text=[f"+{t*100:.1f}%" for t in df_trend["trend"]],
    textposition='outside'
))

fig_trend.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font={'color': 'white'},
    height=400,
    xaxis_title="City",
    yaxis_title="Year-over-Year Change (%)",
    showlegend=False
)

st.plotly_chart(fig_trend, use_container_width=True)

st.divider()

# Rent calculator
st.subheader("🧮 Quick Rent Estimator")

calc_col1, calc_col2, calc_col3 = st.columns(3)

with calc_col1:
    selected_city = st.selectbox(
        "Select City",
        options=list(MARKET_DATA.keys()),
        index=3  # Berlin
    )

with calc_col2:
    apartment_size = st.number_input(
        "Apartment Size (m²)",
        min_value=20,
        max_value=200,
        value=60,
        step=5
    )

with calc_col3:
    # Calculate estimate
    city_data = MARKET_DATA[selected_city]
    base_rent = city_data['avg_rent']
    
    # Size adjustment (larger = slightly lower per sqm)
    if apartment_size > 100:
        size_factor = 0.92
    elif apartment_size > 80:
        size_factor = 0.96
    elif apartment_size < 40:
        size_factor = 1.05
    else:
        size_factor = 1.0
    
    adjusted_rent = base_rent * size_factor
    total_rent = adjusted_rent * apartment_size
    
    st.metric(
        "Estimated Rent",
        f"€{total_rent:.0f}/month",
        delta=f"€{adjusted_rent:.2f}/m²"
    )

# Show range
low_rent = total_rent * 0.85
high_rent = total_rent * 1.15

st.markdown(f"""
<div style="background: linear-gradient(135deg, rgba(0,212,255,0.1), rgba(123,44,191,0.1));
            padding: 20px; border-radius: 12px; text-align: center;">
    <p style="margin: 0;">Typical range for {apartment_size}m² in {selected_city}:</p>
    <h3 style="margin: 10px 0;">€{low_rent:.0f} - €{high_rent:.0f} / month</h3>
    <p style="color: #aaa; margin: 0;">Based on current market data</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# Scam awareness
st.subheader("⚠️ Scam Awareness")

st.markdown("""
<div style="background: rgba(211, 47, 47, 0.1); border-left: 4px solid #d32f2f;
            padding: 20px; border-radius: 8px; margin: 20px 0;">
    <h4>🚨 Common Rental Scams in Germany</h4>
    <ul>
        <li><strong>Advance Payment Scams:</strong> Requests for deposit/rent before viewing</li>
        <li><strong>Fake Landlords:</strong> Claiming to be abroad, unable to meet</li>
        <li><strong>Too-Good-To-Be-True:</strong> Prices 40-60% below market rate</li>
        <li><strong>Key by Mail:</strong> Offering to send keys after payment</li>
        <li><strong>Urgency Tactics:</strong> Pressure to decide immediately</li>
    </ul>
</div>
""", unsafe_allow_html=True)

scam_col1, scam_col2 = st.columns(2)

with scam_col1:
    st.markdown("""
    **🛡️ Protect Yourself:**
    - Never pay before viewing in person
    - Verify landlord identity (ID, ownership docs)
    - Use official platforms with buyer protection
    - Be suspicious of prices far below market
    - Meet in person at the actual apartment
    """)

with scam_col2:
    st.markdown("""
    **📞 Report Scams To:**
    - Local police (Polizei)
    - Consumer protection (Verbraucherzentrale)
    - Platform where listing was found
    - Federal Network Agency (if online fraud)
    """)

st.divider()

# Data disclaimer
st.caption("""
📊 **Data Sources & Disclaimer**: Market data is based on aggregated rental market 
statistics and may not reflect real-time prices. Actual rents can vary significantly 
based on specific location, apartment condition, amenities, and current market conditions. 
Use RentShield's predictions as a general guide, not definitive pricing.
""")
