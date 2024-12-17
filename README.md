# crypto_price_prediction
Project Title: Crypto Price Prediction Model

Overview

This project aims to develop a machine learning model capable of predicting the Adjusted Close price of a cryptocurrency, specifically Bitcoin, based on historical data. The model will utilize a time series analysis approach to capture the temporal dependencies inherent in cryptocurrency price movements.

Data

The dataset used for training and testing the model was sourced from kaggle "mianbilal12/bitcoin-historical-data". The dataset will include historical data for Bitcoin, encompassing features like:

Open Price: The price at the beginning of the trading day.
High Price: The highest price reached during the trading day.
Low Price: The lowest price reached during the trading day.
Close Price: The price at the end of the trading day.
Volume: The trading volume for the day.
Adjusted Close Price: The closing price adjusted for corporate actions like dividends and stock splits.
Data Preprocessing

Data Cleaning:
Handle missing values (e.g., imputation or removal).
Identify and address outliers.
Feature Engineering:
Create relevant features based on domain knowledge and statistical analysis.
Consider technical indicators like Moving Averages, Relative Strength Index (RSI), and Bollinger Bands.
Data Splitting:
Divide the dataset into training and testing sets.
Model Selection and Training

Baseline Model:
Linear Regression Model

Additional Models:
Random Forest
Gradient boosting

Model Training:
Train the selected model on the training dataset.
Experiment with different hyperparameters to optimize performance.
Model Evaluation

Performance Metrics:
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
Mean Absolute Error (MAE)
R-squared

Backtesting:
Evaluate the model's performance on historical data to assess its predictive accuracy.


Deployment

Model Serialization:
Saved the trained model for future use as a pickle file in the directory "model_management_scripts\mlruns\models\BestModel\model.pkl"

Deployment Platform:
API Integration:
Created an API to expose the model's predictions to other applications or services.

Streamlit deployment:
Model was deployed to streamlit. Visit "https://bitcoin-prediction-dep.streamlit.app/" to interact with model

Future Work

Ensemble Methods: Combine multiple models to improve prediction accuracy.
Feature Engineering: Explore additional features and techniques to enhance model performance.
Hyperparameter Tuning: Utilize advanced techniques like Grid Search and Randomized Search to fine-tune hyperparameters.
Real-time Predictions: Implement a system to continuously update the model with new data and generate real-time predictions.
Note:

Remember that cryptocurrency markets are highly volatile and subject to various factors, including economic conditions, geopolitical events, and market sentiment. While this model can provide valuable insights, it's essential to use it as a tool for informed decision-making, not as a guaranteed prediction of future prices.
