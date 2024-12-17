import streamlit as st
import numpy as np
import pickle  # For loading the trained model

# 1. Function to load the pretrained model
def load_trained_model(model_path):
    """
    Load the pretrained model from a file using pickle.
    Args:
        model_path (str): Path to the saved model file.
    Returns:
        model: The loaded trained model.
    """
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# 2. Streamlit Interface
def streamlit_interface():
    # Load the trained model
    model_path = "model_management_scripts/mlruns/models/BestModel" 
    model = load_trained_model(model_path)

    # Streamlit App Layout
    st.set_page_config(page_title="Bitcoin Price Prediction", layout="wide", initial_sidebar_state="expanded")
    
    # Add custom CSS for styling
    st.markdown(
        """
        <style>
            .header {
                color: #4CAF50;
                font-size: 64px;
                font-weight: bold;
                text-align: center;
                margin-bottom: 40px;
            }
            .subheader {
                font-size: 18px;
                font-weight: bold;
                color: #333333;
            }
            .input-field {
                margin-bottom: 20px;
            }
            .prediction-result {
                font-size: 24px;
                font-weight: bold;
                color: #FF5733;
                margin-top: 30px;
            }
            .button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 16px;
            }
            .button:hover {
                background-color: #45a049;
            }
        </style>
        """, unsafe_allow_html=True)

    # App title
    st.markdown('<p class="header">Bitcoin Price Prediction (Using Trained Model)</p>', unsafe_allow_html=True)

    st.subheader("Enter the features for the previous day:")

    # User input fields for the features with improved design
    volume_lag_1 = st.number_input("Previous Day's Volume (Volume_Lag_1)", min_value=0.0, step=1.0, format="%.2f", key="volume_lag_1", help="Enter the previous day's volume.")
    daily_return = st.slider("Previous Day's Daily Return (Daily_Return)", min_value=-1.0, max_value=1.0, step=0.01, key="daily_return", help="Enter the previous day's daily return.")
    avg_adj_close = st.number_input("30-Day Average Adjusted Close (A_30_day_avg_Adj_Close)", min_value=0.0, step=0.01, key="avg_adj_close", help="Enter the 30-day average adjusted close.")
    volatility = st.number_input("30-Day Volatility (A_30_day_volatility)", min_value=0.0, step=0.01, key="volatility", help="Enter the 30-day volatility.")
    
    # Date Picker for Year, Month, Day
    date_input = st.date_input("Select Date", key="date", help="Select the date for prediction.")
    year, month, day = date_input.year, date_input.month, date_input.day

    # Prediction button with improved styling
    predict_button = st.button("Predict Adjusted Close for Next Day", key="predict", help="Click to predict the next day's adjusted close.", use_container_width=True)

    if predict_button:
        with st.spinner("Making the prediction..."):
            # Create the input features array for prediction
            input_features = np.array([[volume_lag_1, daily_return, avg_adj_close, volatility, year, month, day]])

            # Predict the adjusted close price using the trained model
            predicted_adj_close = model.predict(input_features)

            # Display the predicted adjusted close price with improved styling
            st.markdown(f'<p class="prediction-result">Predicted Adjusted Close for Next Day: ${predicted_adj_close[0]:.2f}</p>', unsafe_allow_html=True)

# Run the Streamlit interface
if __name__ == "__main__":
    streamlit_interface()
