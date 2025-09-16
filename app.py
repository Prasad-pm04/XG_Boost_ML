# app.py
import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# === PAGE CONFIGURATION ===
st.set_page_config(
    page_title="Possum Age Predictor | TrainWithPrasadPM",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === CUSTOM CSS STYLING ===
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 0;
    }
    
    .creator-tag {
        background: linear-gradient(45deg, #ff6b6b, #ffa726);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
        margin-top: 1rem;
        font-size: 0.9rem;
    }
    
    .measurement-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .result-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .result-age {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .info-box {
        background: #f8f9ff;
        border: 1px solid #e1e5fe;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    
    .sidebar .stSelectbox > div > div {
        background-color: #f8f9ff;
        border-radius: 10px;
    }
    
    .measurement-group {
        background: #f8f9ff;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border: 1px solid #e1e5fe;
    }
    
    .group-title {
        color: #667eea;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# === LOAD MODEL WITH ERROR HANDLING ===
@st.cache_resource
def load_model():
    try:
        with open("best_possum_rf_nocase.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("❌ Model file not found! Please ensure 'best_possum_rf_nocase.pkl' is in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

model = load_model()

# === HEADER SECTION ===
st.markdown("""
<div class="main-header">
    <h1>🐾 Possum Age Predictor</h1>
    <p>Advanced Machine Learning Model for Wildlife Research</p>
    <div class="creator-tag">
        Created by TrainWithPrasadPM
    </div>
</div>
""", unsafe_allow_html=True)

# === SIDEBAR CONFIGURATION ===
with st.sidebar:
    st.markdown("### 🔧 Model Configuration")
    
    # Population and Sex
    st.markdown('<div class="group-title">🌍 Basic Information</div>', unsafe_allow_html=True)
    Pop = st.selectbox(
        "Population Origin",
        ["Vic", "other"],
        help="Select the geographical population of the possum"
    )
    
    sex = st.selectbox(
        "Sex",
        ["m", "f"],
        format_func=lambda x: "Male 🐾" if x == "m" else "Female 🐾",
        help="Select the sex of the possum"
    )
    
    # Model Info
    st.markdown("---")
    st.markdown("### 📊 Model Information")
    st.info("**Model Type:** XGBoost Regressor\n\n**Features:** 11 morphological measurements\n\n**Accuracy:** Optimized for wildlife research")
    
    # Instructions
    st.markdown("### 📋 Instructions")
    st.markdown("""
    1. Enter all physical measurements
    2. Select population and sex
    3. Click 'Predict Age' button
    4. Review the estimated age
    
    **Note:** All measurements should be in millimeters.
    """)

# === MAIN CONTENT AREA ===
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📏 Physical Measurements")
    
    # Head Measurements Group
    st.markdown('<div class="measurement-group">', unsafe_allow_html=True)
    st.markdown('<div class="group-title">🦴 Head & Skull Measurements</div>', unsafe_allow_html=True)
    
    col_head1, col_head2 = st.columns(2)
    with col_head1:
        hdlngth = st.number_input(
            "Head Length (mm)",
            min_value=50.0,
            max_value=120.0,
            value=90.0,
            step=0.1,
            help="Length of the head from tip of nose to back of skull"
        )
    
    with col_head2:
        skullw = st.number_input(
            "Skull Width (mm)",
            min_value=40.0,
            max_value=80.0,
            value=60.0,
            step=0.1,
            help="Width of the skull at its widest point"
        )
    
    eye = st.number_input(
        "Eye Size (mm)",
        min_value=10.0,
        max_value=25.0,
        value=15.0,
        step=0.1,
        help="Diameter of the eye"
    )
    
    earconch = st.number_input(
        "Ear Conch Length (mm)",
        min_value=40.0,
        max_value=70.0,
        value=55.0,
        step=0.1,
        help="Length of the external ear"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Body Measurements Group
    st.markdown('<div class="measurement-group">', unsafe_allow_html=True)
    st.markdown('<div class="group-title">🦘 Body Measurements</div>', unsafe_allow_html=True)
    
    col_body1, col_body2 = st.columns(2)
    with col_body1:
        totlngth = st.number_input(
            "Total Length (mm)",
            min_value=60.0,
            max_value=120.0,
            value=90.0,
            step=0.1,
            help="Total body length from nose to base of tail"
        )
    
    with col_body2:
        taill = st.number_input(
            "Tail Length (mm)",
            min_value=20.0,
            max_value=60.0,
            value=35.0,
            step=0.1,
            help="Length of the tail"
        )
    
    footlgth = st.number_input(
        "Foot Length (mm)",
        min_value=50.0,
        max_value=90.0,
        value=70.0,
        step=0.1,
        help="Length of the hind foot"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Girth Measurements Group
    st.markdown('<div class="measurement-group">', unsafe_allow_html=True)
    st.markdown('<div class="group-title">📐 Girth Measurements</div>', unsafe_allow_html=True)
    
    col_girth1, col_girth2 = st.columns(2)
    with col_girth1:
        chest = st.number_input(
            "Chest Girth (mm)",
            min_value=20.0,
            max_value=40.0,
            value=30.0,
            step=0.1,
            help="Circumference around the chest"
        )
    
    with col_girth2:
        belly = st.number_input(
            "Belly Girth (mm)",
            min_value=20.0,
            max_value=50.0,
            value=35.0,
            step=0.1,
            help="Circumference around the belly"
        )
    st.markdown('</div>', unsafe_allow_html=True)

# === RIGHT COLUMN - VISUALIZATION ===
with col2:
    st.markdown("### 📊 Measurement Summary")
    
    # Create a summary visualization
    measurements_data = {
        'Measurement': ['Head Length', 'Skull Width', 'Total Length', 'Tail Length', 
                       'Foot Length', 'Ear Conch', 'Eye Size', 'Chest Girth', 'Belly Girth'],
        'Value': [hdlngth, skullw, totlngth, taill, footlgth, earconch, eye, chest, belly],
        'Category': ['Head', 'Head', 'Body', 'Body', 'Body', 'Head', 'Head', 'Body', 'Body']
    }
    
    df_viz = pd.DataFrame(measurements_data)
    
    # Create a horizontal bar chart
    fig = px.bar(df_viz, 
                 x='Value', 
                 y='Measurement', 
                 color='Category',
                 orientation='h',
                 color_discrete_map={'Head': '#667eea', 'Body': '#764ba2'},
                 title="Current Measurements Overview")
    
    fig.update_layout(
        height=400,
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title_font_color='#333',
        font_color='#333'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Quick stats
    st.markdown("### 📈 Quick Statistics")
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric("Population", "Victoria" if Pop == "Vic" else "Other")
    with col_stat2:
        st.metric("Sex", "Male" if sex == "m" else "Female")

# === PREDICTION SECTION ===
st.markdown("---")

# Center the prediction button
col_pred1, col_pred2, col_pred3 = st.columns([1, 1, 1])
with col_pred2:
    predict_button = st.button("🔮 Predict Possum Age", use_container_width=True)

if predict_button:
    # Prepare input for model
    input_dict = {
        "hdlngth": hdlngth,
        "skullw": skullw,
        "totlngth": totlngth,
        "taill": taill,
        "footlgth": footlgth,
        "earconch": earconch,
        "eye": eye,
        "chest": chest,
        "belly": belly,
        "Pop_Vic": 1 if Pop == "Vic" else 0,
        "sex_m": 1 if sex == "m" else 0
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # Add loading animation
    with st.spinner('🧠 Analyzing possum characteristics...'):
        time.sleep(1)  # Add slight delay for UX
        prediction = model.predict(input_df)[0]
    
    # Display result with enhanced styling
    st.markdown(f"""
    <div class="result-card">
        <h2>🎯 Prediction Result</h2>
        <div class="result-age">{prediction:.1f}</div>
        <h3>Estimated Years Old</h3>
        <p>Based on the provided morphological measurements</p>
        <small>Prediction generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Additional insights
    st.markdown("### 🔍 Prediction Insights")
    
    col_insight1, col_insight2, col_insight3 = st.columns(3)
    
    with col_insight1:
        age_category = "Young Adult" if prediction < 3 else "Adult" if prediction < 6 else "Senior"
        st.info(f"**Age Category:** {age_category}")
    
    with col_insight2:
        confidence = "High" if 1 <= prediction <= 8 else "Medium"
        st.info(f"**Model Confidence:** {confidence}")
    
    with col_insight3:
        population_info = "Victoria, Australia" if Pop == "Vic" else "Other Region"
        st.info(f"**Population:** {population_info}")

# === FOOTER ===
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666; background: #f8f9ff; border-radius: 10px; margin-top: 2rem;">
    <h4>🐾 Possum Age Predictor</h4>
    <p>Developed by <strong>TrainWithPrasadPM</strong> | Powered by XGBoost Machine Learning</p>
    <p><small>For wildlife research and conservation purposes</small></p>
</div>
""", unsafe_allow_html=True)
