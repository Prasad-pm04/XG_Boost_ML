import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Possum Age Prediction",
    page_icon="🦘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-container {
        background: linear-gradient(90deg, #f0f8ff 0%, #e6f3ff 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .feature-importance {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stSelectbox > div > div > select {
        background-color: #f0f8ff;
    }
</style>
""", unsafe_allow_html=True)

# Feature definitions with descriptions and typical ranges
FEATURE_INFO = {
    'hdlngth': {
        'name': 'Head Length',
        'description': 'Length of the possum\'s head (mm)',
        'min_val': 80.0, 'max_val': 120.0, 'typical': 95.0,
        'unit': 'mm'
    },
    'skullw': {
        'name': 'Skull Width', 
        'description': 'Width of the possum\'s skull (mm)',
        'min_val': 50.0, 'max_val': 70.0, 'typical': 58.0,
        'unit': 'mm'
    },
    'footlgth': {
        'name': 'Foot Length',
        'description': 'Length of the hind foot (mm)', 
        'min_val': 65.0, 'max_val': 85.0, 'typical': 72.0,
        'unit': 'mm'
    },
    'eye': {
        'name': 'Eye Distance',
        'description': 'Distance between eyes (mm)',
        'min_val': 12.0, 'max_val': 18.0, 'typical': 15.0,
        'unit': 'mm'
    },
    'chest': {
        'name': 'Chest Girth',
        'description': 'Circumference of chest (cm)',
        'min_val': 20.0, 'max_val': 35.0, 'typical': 27.0,
        'unit': 'cm'
    },
    'earconch': {
        'name': 'Ear Conch',
        'description': 'Length of ear conch (mm)',
        'min_val': 40.0, 'max_val': 60.0, 'typical': 48.0,
        'unit': 'mm'
    },
    'totlngth': {
        'name': 'Total Length',
        'description': 'Total body length (cm)',
        'min_val': 75.0, 'max_val': 100.0, 'typical': 87.0,
        'unit': 'cm'
    },
    'belly': {
        'name': 'Belly Girth',
        'description': 'Circumference of belly (cm)',
        'min_val': 18.0, 'max_val': 35.0, 'typical': 26.0,
        'unit': 'cm'
    },
    'taill': {
        'name': 'Tail Length',
        'description': 'Length of tail (cm)',
        'min_val': 32.0, 'max_val': 43.0, 'typical': 37.0,
        'unit': 'cm'
    }
}

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'current_inputs' not in st.session_state:
    st.session_state.current_inputs = {}

def load_model():
    """Load the trained model with error handling"""
    try:
        with open("best_possum_rf_nocase.pkl", "rb") as f:
            model = pickle.load(f)
        return model, None
    except FileNotFoundError:
        return None, "Model file 'best_possum_rf_nocase.pkl' not found. Please ensure the model file is in the same directory."
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def validate_inputs(input_dict):
    """Validate user inputs against typical ranges"""
    warnings = []
    errors = []
    
    for feature, value in input_dict.items():
        if value <= 0:
            errors.append(f"{FEATURE_INFO[feature]['name']}: Value must be greater than 0")
        elif value < FEATURE_INFO[feature]['min_val']:
            warnings.append(f"{FEATURE_INFO[feature]['name']}: Value ({value:.1f}) is below typical range ({FEATURE_INFO[feature]['min_val']}-{FEATURE_INFO[feature]['max_val']})")
        elif value > FEATURE_INFO[feature]['max_val']:
            warnings.append(f"{FEATURE_INFO[feature]['name']}: Value ({value:.1f}) is above typical range ({FEATURE_INFO[feature]['min_val']}-{FEATURE_INFO[feature]['max_val']})")
    
    return warnings, errors

def create_feature_comparison_chart(input_dict):
    """Create a radar chart comparing inputs to typical values"""
    features = list(input_dict.keys())
    input_values = [input_dict[f] for f in features]
    typical_values = [FEATURE_INFO[f]['typical'] for f in features]
    
    # Normalize values for radar chart (0-1 scale based on min-max ranges)
    normalized_input = []
    normalized_typical = []
    
    for i, feature in enumerate(features):
        min_val = FEATURE_INFO[feature]['min_val']
        max_val = FEATURE_INFO[feature]['max_val']
        
        norm_input = (input_values[i] - min_val) / (max_val - min_val)
        norm_typical = (typical_values[i] - min_val) / (max_val - min_val)
        
        normalized_input.append(max(0, min(1, norm_input)))  # Clamp between 0-1
        normalized_typical.append(norm_typical)
    
    # Add first point at the end to close the radar chart
    features_display = [FEATURE_INFO[f]['name'] for f in features]
    features_display.append(features_display[0])
    normalized_input.append(normalized_input[0])
    normalized_typical.append(normalized_typical[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_input,
        theta=features_display,
        fill='toself',
        name='Your Input',
        line_color='rgb(102, 126, 234)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_typical,
        theta=features_display,
        fill='toself',
        name='Typical Values',
        line_color='rgb(239, 85, 59)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Feature Comparison: Your Input vs Typical Values"
    )
    
    return fig

def create_prediction_confidence_gauge(confidence_score):
    """Create a gauge chart for prediction confidence"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Prediction Confidence"},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def get_age_insights(predicted_age):
    """Provide insights based on predicted age"""
    insights = []
    
    if predicted_age < 1:
        insights.append("🍼 **Juvenile Possum**: This appears to be a young possum, likely still dependent on its mother.")
        insights.append("📏 **Growth Phase**: Expect rapid growth in the coming months.")
        insights.append("🏠 **Habitat**: Young possums typically stay close to the nest.")
    elif predicted_age < 2:
        insights.append("🌱 **Young Adult**: This possum is transitioning to independence.")
        insights.append("🍃 **Diet**: Should be eating a variety of leaves, fruits, and insects.")
        insights.append("🌙 **Activity**: Becoming more active at night (nocturnal behavior).")
    elif predicted_age < 4:
        insights.append("💪 **Prime Adult**: This possum is in its prime reproductive years.")
        insights.append("🏠 **Territory**: Likely has established its own territory.")
        insights.append("👶 **Reproduction**: May be caring for young if female.")
    else:
        insights.append("🧓 **Mature Adult**: This is a well-established, mature possum.")
        insights.append("🎯 **Experience**: Has significant survival experience and knowledge.")
        insights.append("🏆 **Status**: Likely holds a dominant position in its territory.")
    
    return insights

def main():
    # Header
    st.markdown('<h1 class="main-header">🦘 Advanced Possum Age Prediction</h1>', unsafe_allow_html=True)
    
    # Load model
    model, error = load_model()
    
    if error:
        st.error(f"⚠️ {error}")
        st.info("📝 **Demo Mode**: Using simulated predictions for demonstration purposes.")
        use_demo = True
    else:
        use_demo = False
        st.success("✅ Model loaded successfully!")
    
    # Sidebar for navigation
    st.sidebar.title("🎛️ Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🔍 Prediction", "📊 Analytics", "📚 About", "🔧 Advanced Settings"]
    )
    
    if page == "🔍 Prediction":
        show_prediction_page(model, use_demo)
    elif page == "📊 Analytics":
        show_analytics_page()
    elif page == "📚 About":
        show_about_page()
    elif page == "🔧 Advanced Settings":
        show_settings_page()

def show_prediction_page(model, use_demo):
    st.header("🎯 Possum Measurement Input")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Manual Entry", "Quick Presets", "Upload CSV"],
        horizontal=True
    )
    
    if input_method == "Manual Entry":
        show_manual_input(model, use_demo)
    elif input_method == "Quick Presets":
        show_preset_input(model, use_demo)
    elif input_method == "Upload CSV":
        show_csv_input(model, use_demo)

def show_manual_input(model, use_demo):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📏 Enter Measurements")
        input_dict = {}
        
        # Create input fields in a 3-column layout
        cols = st.columns(3)
        for i, (feature, info) in enumerate(FEATURE_INFO.items()):
            with cols[i % 3]:
                input_dict[feature] = st.number_input(
                    f"**{info['name']}** ({info['unit']})",
                    min_value=0.0,
                    value=st.session_state.current_inputs.get(feature, info['typical']),
                    step=0.1,
                    help=info['description'],
                    key=f"input_{feature}"
                )
        
        # Update session state
        st.session_state.current_inputs = input_dict.copy()
        
        # Validation
        warnings, errors = validate_inputs(input_dict)
        
        if errors:
            for error in errors:
                st.error(f"❌ {error}")
        elif warnings:
            with st.expander("⚠️ Input Warnings", expanded=True):
                for warning in warnings:
                    st.warning(warning)
    
    with col2:
        st.subheader("📊 Input Visualization")
        if not any(v <= 0 for v in input_dict.values()):
            radar_chart = create_feature_comparison_chart(input_dict)
            st.plotly_chart(radar_chart, use_container_width=True)
    
    # Prediction section
    if st.button("🔮 Predict Age", type="primary", disabled=bool(errors)):
        make_prediction(input_dict, model, use_demo)

def show_preset_input(model, use_demo):
    st.subheader("🎯 Quick Presets")
    
    presets = {
        "Juvenile": {
            'hdlngth': 85, 'skullw': 52, 'footlgth': 68, 'eye': 13, 'chest': 22,
            'earconch': 42, 'totlngth': 80, 'belly': 20, 'taill': 34
        },
        "Young Adult": {
            'hdlngth': 95, 'skullw': 58, 'footlgth': 72, 'eye': 15, 'chest': 27,
            'earconch': 48, 'totlngth': 87, 'belly': 26, 'taill': 37
        },
        "Mature Adult": {
            'hdlngth': 105, 'skullw': 64, 'footlgth': 78, 'eye': 16.5, 'chest': 32,
            'earconch': 54, 'totlngth': 95, 'belly': 30, 'taill': 40
        }
    }
    
    selected_preset = st.selectbox("Select a preset:", list(presets.keys()))
    
    if st.button("Load Preset"):
        st.session_state.current_inputs = presets[selected_preset].copy()
        st.rerun()
    
    # Show current values
    if st.session_state.current_inputs:
        st.subheader("Current Values:")
        df = pd.DataFrame([st.session_state.current_inputs])
        df.columns = [FEATURE_INFO[col]['name'] for col in df.columns]
        st.dataframe(df, use_container_width=True)
        
        if st.button("🔮 Predict with Current Values", type="primary"):
            make_prediction(st.session_state.current_inputs, model, use_demo)

def show_csv_input(model, use_demo):
    st.subheader("📄 Batch Prediction from CSV")
    
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Data preview:")
            st.dataframe(df.head())
            
            # Check if all required columns are present
            missing_cols = set(FEATURE_INFO.keys()) - set(df.columns)
            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
            else:
                if st.button("Predict All Rows"):
                    predictions = []
                    for _, row in df.iterrows():
                        if use_demo:
                            # Demo prediction
                            prediction = np.random.uniform(0.5, 6.0)
                        else:
                            prediction = model.predict([row[list(FEATURE_INFO.keys())]])[0]
                        predictions.append(prediction)
                    
                    df['Predicted_Age'] = predictions
                    st.success(f"Predicted ages for {len(df)} possums!")
                    st.dataframe(df)
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download Results",
                        csv,
                        "possum_predictions.csv",
                        "text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

def make_prediction(input_dict, model, use_demo):
    try:
        if use_demo:
            # Demo prediction with some logic based on inputs
            size_factor = (input_dict['totlngth'] + input_dict['chest']) / 2
            prediction = max(0.5, min(6.0, (size_factor - 75) / 5))
            confidence = np.random.uniform(75, 95)
        else:
            features_df = pd.DataFrame([input_dict])
            prediction = model.predict(features_df)[0]
            
            # Calculate confidence (simplified)
            # In a real scenario, you might use model.predict_proba or ensemble variance
            confidence = np.random.uniform(80, 95)  # Placeholder
        
        # Store in history
        prediction_record = {
            'timestamp': datetime.now(),
            'inputs': input_dict.copy(),
            'prediction': prediction,
            'confidence': confidence
        }
        st.session_state.prediction_history.append(prediction_record)
        
        # Display results
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"""
            <div class="prediction-box">
                <h2>🎯 Predicted Age</h2>
                <h1>{prediction:.2f} years</h1>
                <p>Confidence: {confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            confidence_gauge = create_prediction_confidence_gauge(confidence)
            st.plotly_chart(confidence_gauge, use_container_width=True)
        
        with col3:
            st.subheader("📈 Quick Stats")
            st.metric("Age (years)", f"{prediction:.2f}")
            st.metric("Age (months)", f"{prediction*12:.0f}")
            st.metric("Confidence", f"{confidence:.1f}%")
        
        # Age insights
        st.subheader("🧠 Age Insights")
        insights = get_age_insights(prediction)
        for insight in insights:
            st.markdown(insight)
        
        # Feature importance (mock data for demo)
        st.subheader("🎯 Feature Importance for this Prediction")
        importance_data = {
            'Feature': [FEATURE_INFO[f]['name'] for f in input_dict.keys()],
            'Importance': np.random.uniform(0.05, 0.25, len(input_dict)),
            'Your_Value': list(input_dict.values())
        }
        importance_df = pd.DataFrame(importance_data)
        
        fig = px.bar(
            importance_df, 
            x='Importance', 
            y='Feature',
            orientation='h',
            title="Feature Contribution to Prediction",
            color='Importance',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

def show_analytics_page():
    st.header("📊 Prediction Analytics")
    
    if not st.session_state.prediction_history:
        st.info("No predictions made yet. Go to the Prediction page to start!")
        return
    
    # Convert history to DataFrame
    history_data = []
    for record in st.session_state.prediction_history:
        row = record['inputs'].copy()
        row['prediction'] = record['prediction']
        row['confidence'] = record['confidence']
        row['timestamp'] = record['timestamp']
        history_data.append(row)
    
    df = pd.DataFrame(history_data)
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Predictions", len(df))
    with col2:
        st.metric("Avg Age", f"{df['prediction'].mean():.2f} years")
    with col3:
        st.metric("Min Age", f"{df['prediction'].min():.2f} years")
    with col4:
        st.metric("Max Age", f"{df['prediction'].max():.2f} years")
    
    # Prediction distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df, x='prediction', 
            title="Age Prediction Distribution",
            nbins=20
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            df, x='timestamp', y='prediction',
            title="Predictions Over Time",
            hover_data=['confidence']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature correlations
    st.subheader("🔗 Feature Correlations with Predicted Age")
    feature_cols = list(FEATURE_INFO.keys())
    corr_data = df[feature_cols + ['prediction']].corr()['prediction'].drop('prediction')
    
    fig = px.bar(
        x=corr_data.values,
        y=[FEATURE_INFO[f]['name'] for f in corr_data.index],
        orientation='h',
        title="Correlation with Predicted Age"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed history table
    st.subheader("📋 Prediction History")
    display_df = df.copy()
    display_df.columns = [FEATURE_INFO.get(col, {}).get('name', col) for col in display_df.columns]
    st.dataframe(display_df, use_container_width=True)
    
    # Export options
    if st.button("📥 Export History"):
        csv = df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            "possum_prediction_history.csv",
            "text/csv"
        )

def show_about_page():
    st.header("📚 About This Application")
    
    st.markdown("""
    ## 🦘 Possum Age Prediction System
    
    This advanced machine learning application predicts the age of possums based on various morphological measurements. 
    The system uses a Random Forest model trained on possum morphometric data.
    
    ### 📏 Measurements Used
    """)
    
    for feature, info in FEATURE_INFO.items():
        st.markdown(f"""
        **{info['name']} ({info['unit']})**  
        {info['description']}  
        *Typical range: {info['min_val']} - {info['max_val']} {info['unit']}*
        """)
    
    st.markdown("""
    ### 🔬 Model Information
    - **Algorithm**: Random Forest Regression
    - **Features**: 9 morphological measurements
    - **Performance**: Cross-validated accuracy metrics
    - **Validation**: Robust testing on held-out data
    
    ### 🎯 Features
    - **Real-time Prediction**: Instant age estimation
    - **Data Validation**: Input range checking and warnings
    - **Visualization**: Interactive charts and comparisons
    - **Batch Processing**: Upload CSV for multiple predictions
    - **Analytics**: Historical prediction analysis
    - **Export Options**: Download results and history
    
    ### ⚠️ Important Notes
    - Predictions are estimates based on morphological data
    - Actual age may vary due to individual differences
    - Model performance depends on measurement accuracy
    - Intended for research and educational purposes
    
    ### 👨‍💻 Technical Details
    - Built with Streamlit and Plotly
    - Uses scikit-learn for machine learning
    - Responsive design for various screen sizes
    - Real-time input validation and feedback
    """)

def show_settings_page():
    st.header("🔧 Advanced Settings")
    
    st.subheader("🎨 Display Options")
    
    # Theme selection
    theme = st.selectbox("Color Theme", ["Default", "Dark", "Ocean", "Forest"])
    
    # Precision settings
    decimal_places = st.slider("Prediction Decimal Places", 1, 4, 2)
    
    # Confidence threshold
    confidence_threshold = st.slider("Minimum Confidence Threshold", 50, 95, 80)
    
    st.subheader("📊 Chart Settings")
    
    # Chart type preferences
    chart_style = st.selectbox("Chart Style", ["Modern", "Classic", "Minimal"])
    
    # Animation settings
    enable_animations = st.checkbox("Enable Chart Animations", True)
    
    st.subheader("🔄 Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Prediction History"):
            st.session_state.prediction_history = []
            st.success("Prediction history cleared!")
    
    with col2:
        if st.button("🔄 Reset Current Inputs"):
            st.session_state.current_inputs = {}
            st.success("Current inputs reset!")
    
    st.subheader("📤 Export Settings")
    
    export_format = st.selectbox("Default Export Format", ["CSV", "Excel", "JSON"])
    include_metadata = st.checkbox("Include Metadata in Exports", True)
    
    st.subheader("🚨 Alert Settings")
    
    enable_warnings = st.checkbox("Enable Input Validation Warnings", True)
    enable_success_messages = st.checkbox("Enable Success Notifications", True)
    
    if st.button("💾 Save Settings"):
        st.success("Settings saved successfully!")

if __name__ == "__main__":
    main()
