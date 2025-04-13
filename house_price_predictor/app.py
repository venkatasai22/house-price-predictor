import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load models and label encoders
try:
    linear_model = joblib.load('linear_model.joblib')
    rf_model = joblib.load('rf_model.joblib')
    le_location = joblib.load('label_encoder_location.joblib')
    le_furnishing = joblib.load('label_encoder_furnishing.joblib')
except:
    st.error("Models not found. Please run train_model.py first.")
    st.stop()

# App title
st.title('House Price Predictor')

# Sidebar for additional options
with st.sidebar:
    st.header("Options")
    uploaded_file = st.file_uploader("Upload custom dataset (CSV)", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Dataset uploaded successfully!")
            st.write(f"Records: {len(df)}")
        except Exception as e:
            st.error(f"Error loading file: {e}")

# Main prediction form
with st.form("prediction_form"):
    st.header("Property Details")
    
    col1, col2 = st.columns(2)
    with col1:
        bhk = st.number_input("BHK", min_value=1, max_value=10, value=2)
        area = st.number_input("Area (sq ft)", min_value=300, max_value=10000, value=1000)
        floors = st.number_input("Floors", min_value=1, max_value=10, value=1)
        
    with col2:
        location = st.selectbox("Location", options=['A', 'B', 'C', 'D', 'E'])
        age = st.number_input("Property Age (years)", min_value=0, max_value=100, value=5)
        furnished = st.selectbox("Furnishing", options=['Furnished', 'Semi-Furnished', 'Unfurnished'])
    
    submitted = st.form_submit_button("Predict Price")
    
    if submitted:
        # Prepare input data
        input_data = pd.DataFrame({
            'BHK': [bhk],
            'Area': [area],
            'Location': [location],
            'Floors': [floors],
            'Age': [age],
            'Furnishing': [furnished]
        })
        
        # Encode categorical features with error handling
        try:
            input_data['Location'] = le_location.transform(input_data['Location'])
        except ValueError:
            st.error("Invalid location selected")
            st.stop()  # Use st.stop() instead of return
            
        try:
            input_data['Furnishing'] = le_furnishing.transform(input_data['Furnishing'])
        except ValueError:
            st.warning("Unknown furnishing type - defaulting to Semi-Furnished")
            input_data['Furnishing'] = le_furnishing.transform(['Semi-Furnished'])[0]
        
        # Make predictions
        linear_pred = linear_model.predict(input_data)[0]
        rf_pred = rf_model.predict(input_data)[0]
        
        # Display results
        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        col1.metric("Linear Regression", f"${linear_pred:,.0f}")
        col2.metric("Random Forest", f"${rf_pred:,.0f}")
        
        # Show comparison chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Model comparison
        models = ['Linear Regression', 'Random Forest']
        prices = [linear_pred, rf_pred]
        sns.barplot(x=models, y=prices, ax=ax1)
        ax1.set_ylabel('Predicted Price')
        ax1.set_title('Model Comparison')
        
        # Feature importance (for Random Forest)
        if hasattr(rf_model, 'feature_importances_'):
            importance = pd.DataFrame({
                'Feature': ['BHK', 'Area', 'Location', 'Floors', 'Age', 'Furnishing'],
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            sns.barplot(x='Importance', y='Feature', data=importance, ax=ax2)
            ax2.set_title('Feature Importance')
        
        st.pyplot(fig)

# Instructions
st.markdown("""
### How It Works
1. Enter property details (BHK, Area, Location)
2. Click "Predict Price"
3. View predictions from both models

Note: This is a demo using sample data. For production use, train with a real housing dataset.
""")
