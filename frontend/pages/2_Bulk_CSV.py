"""
RentShield Germany - Bulk CSV Analysis
Upload and analyze multiple listings at once.
"""

import streamlit as st
import requests
import pandas as pd
import io
import os

# Configuration
API_URL = os.getenv("RENTSHIELD_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Bulk Analysis | RentShield",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .upload-area {
        border: 2px dashed rgba(123, 44, 191, 0.5);
        border-radius: 16px;
        padding: 40px;
        text-align: center;
        background: rgba(123, 44, 191, 0.05);
    }
    
    .stats-card {
        background: linear-gradient(135deg, rgba(0,212,255,0.1), rgba(123,44,191,0.1));
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def get_risk_color(risk_level: str) -> str:
    """Get color for risk level."""
    colors = {
        "LOW": "#00c853",
        "MEDIUM": "#ffc107", 
        "HIGH": "#ff5722",
        "CRITICAL": "#d32f2f"
    }
    return colors.get(risk_level, "#9e9e9e")


def analyze_listings(listings: list) -> dict:
    """Call FastAPI bulk endpoint."""
    try:
        response = requests.post(
            f"{API_URL}/predict/bulk",
            json={"listings": listings},
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None


def process_csv(df: pd.DataFrame) -> list:
    """Convert DataFrame to list of listing dicts."""
    required_cols = ['city', 'postcode', 'living_space', 'rooms', 'rent']
    
    # Check required columns
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return None
    
    listings = []
    for _, row in df.iterrows():
        listing = {
            "city": str(row.get('city', '')),
            "postcode": str(row.get('postcode', '')),
            "living_space": float(row.get('living_space', 50)),
            "rooms": float(row.get('rooms', 2)),
            "rent": float(row.get('rent', 800)),
            "description": str(row.get('description', '')),
            "year_built": int(row['year_built']) if pd.notna(row.get('year_built')) else None,
            "floor": int(row['floor']) if pd.notna(row.get('floor')) else None
        }
        listings.append(listing)
    
    return listings


# Page content
st.title("📊 Bulk CSV Analysis")
st.caption("Upload and analyze multiple listings at once")

st.divider()

# File upload
st.subheader("📁 Upload CSV File")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV with rental listings to analyze"
    )

with col2:
    st.markdown("**Required Columns:**")
    st.code("""
city
postcode
living_space
rooms
rent
    """)
    
    st.markdown("**Optional Columns:**")
    st.code("""
description
year_built
floor
    """)

# Download template
st.divider()
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    template_df = pd.DataFrame({
        'city': ['Berlin', 'Munich', 'Hamburg'],
        'postcode': ['10115', '80331', '20095'],
        'living_space': [55, 70, 45],
        'rooms': [2, 3, 1.5],
        'rent': [950, 1800, 750],
        'description': [
            'Beautiful apartment in Mitte',
            'Luxury flat with balcony',
            'Cozy studio near river'
        ],
        'year_built': [1990, 2015, 1985],
        'floor': [3, 5, 1]
    })
    
    csv_template = template_df.to_csv(index=False)
    st.download_button(
        "📥 Download Template CSV",
        data=csv_template,
        file_name="rentshield_template.csv",
        mime="text/csv",
        use_container_width=True
    )

st.divider()

# Process uploaded file
if uploaded_file is not None:
    # Read CSV
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df)} listings from CSV")
        
        # Preview
        with st.expander("👁️ Preview Data", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Analyze button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_btn = st.button(
                "🔍 Analyze All Listings",
                use_container_width=True,
                type="primary"
            )
        
        if analyze_btn:
            # Convert to listings
            listings = process_csv(df)
            
            if listings:
                with st.spinner(f"Analyzing {len(listings)} listings..."):
                    # Progress bar
                    progress_bar = st.progress(0)
                    
                    # Call API
                    result = analyze_listings(listings)
                    progress_bar.progress(100)
                
                if result:
                    st.divider()
                    st.subheader("📈 Analysis Results")
                    
                    # Summary stats
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Total Analyzed",
                            result['total_processed'],
                            help="Number of listings processed"
                        )
                    
                    with col2:
                        st.metric(
                            "High Risk",
                            result['high_risk_count'],
                            delta=f"{result['high_risk_count']/result['total_processed']*100:.0f}%" if result['total_processed'] > 0 else "0%",
                            delta_color="inverse"
                        )
                    
                    with col3:
                        avg_scam = sum(p['scam_score'] for p in result['predictions']) / len(result['predictions'])
                        st.metric(
                            "Avg Risk Score",
                            f"{avg_scam*100:.0f}%"
                        )
                    
                    with col4:
                        overpriced = sum(1 for p in result['predictions'] if p['is_overpriced'])
                        st.metric(
                            "Overpriced",
                            overpriced,
                            delta=f"{overpriced/result['total_processed']*100:.0f}%" if result['total_processed'] > 0 else "0%",
                            delta_color="inverse"
                        )
                    
                    # Create results dataframe
                    results_data = []
                    for i, (pred, row) in enumerate(zip(result['predictions'], df.to_dict('records'))):
                        results_data.append({
                            'City': row.get('city', ''),
                            'Postcode': row.get('postcode', ''),
                            'Size (m²)': row.get('living_space', 0),
                            'Rent (€)': row.get('rent', 0),
                            'Fair Rent (€)': pred['predicted_rent'],
                            'Overpricing': f"{pred['overpricing_pct']*100:+.1f}%",
                            'Scam Score': f"{pred['scam_score']*100:.0f}%",
                            'Risk Level': pred['risk_level'],
                            'Issues': '; '.join(pred['reasons'][:2]) if pred['reasons'] else 'None'
                        })
                    
                    results_df = pd.DataFrame(results_data)
                    
                    st.divider()
                    
                    # Risk level filter
                    st.subheader("📋 Detailed Results")
                    
                    risk_filter = st.multiselect(
                        "Filter by Risk Level",
                        options=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                        default=["HIGH", "CRITICAL"]
                    )
                    
                    # Filter results
                    if risk_filter:
                        filtered_df = results_df[results_df['Risk Level'].isin(risk_filter)]
                    else:
                        filtered_df = results_df
                    
                    # Display results with color coding
                    def color_risk(val):
                        colors = {
                            'LOW': 'background-color: rgba(0, 200, 83, 0.3)',
                            'MEDIUM': 'background-color: rgba(255, 193, 7, 0.3)',
                            'HIGH': 'background-color: rgba(255, 87, 34, 0.3)',
                            'CRITICAL': 'background-color: rgba(211, 47, 47, 0.3)'
                        }
                        return colors.get(val, '')
                    
                    styled_df = filtered_df.style.applymap(
                        color_risk, 
                        subset=['Risk Level']
                    )
                    
                    st.dataframe(styled_df, use_container_width=True, height=400)
                    
                    st.divider()
                    
                    # Download results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv_results = results_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results CSV",
                            data=csv_results,
                            file_name="rentshield_results.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        # High risk only
                        high_risk_df = results_df[results_df['Risk Level'].isin(['HIGH', 'CRITICAL'])]
                        if not high_risk_df.empty:
                            csv_high_risk = high_risk_df.to_csv(index=False)
                            st.download_button(
                                "🚨 Download High Risk Only",
                                data=csv_high_risk,
                                file_name="rentshield_high_risk.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        else:
                            st.success("No high-risk listings found! 🎉")
    
    except Exception as e:
        st.error(f"Error reading CSV: {str(e)}")
        st.info("Please ensure your CSV is properly formatted with the required columns.")

else:
    # Info when no file uploaded
    st.info("👆 Upload a CSV file to begin bulk analysis")
    
    with st.expander("📖 Instructions"):
        st.markdown("""
        ### How to use Bulk Analysis
        
        1. **Prepare your CSV file** with the required columns
        2. **Upload the file** using the file uploader above
        3. **Review the preview** to ensure data looks correct
        4. **Click Analyze** to process all listings
        5. **Download results** in CSV format
        
        ### CSV Format Example
        
        ```csv
        city,postcode,living_space,rooms,rent,description,year_built,floor
        Berlin,10115,55,2,950,Nice apartment,1990,3
        Munich,80331,70,3,1800,Luxury flat,2015,5
        ```
        
        ### Tips
        
        - Include the description text for better scam detection
        - Use German city names (Berlin, München, etc.)
        - Rent should be cold rent (Kaltmiete) in euros
        - Large files may take longer to process
        """)
