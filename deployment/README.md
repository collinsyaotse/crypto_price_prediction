# Bitcoin Price Prediction - Streamlit App

This repository contains a Streamlit web application for predicting the adjusted close price of Bitcoin using a trained machine learning model. The app uses a set of features from the previous day's data to make predictions about the next day's Bitcoin adjusted close price.

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Running the App](#running-the-app)
4. [Accessing the App](#accessing-the-app)
5. [Usage](#usage)
6. [App Features](#app-features)
7. [Technologies Used](#technologies-used)

## Overview

This Streamlit app provides a user-friendly interface to predict the adjusted close price of Bitcoin based on the following input features:

- **Previous Day's Volume (Volume_Lag_1)**
- **Previous Day's Daily Return (Daily_Return)**
- **30-Day Average Adjusted Close (A_30_day_avg_Adj_Close)**
- **30-Day Volatility (A_30_day_volatility)**
- **Year**
- **Month**
- **Day**

The trained model is downloaded from GitHub and used to make predictions.

## Installation

To run this app locally, follow these steps:

1. Visit https://cryptopriceprediction-capstone.streamlit.app/ and interact with app
