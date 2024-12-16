import os
import pandas as pd
import mlflow
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from prefect import flow, task
import kagglehub  

# 1. Ingest Data Task
@task
def ingest_data():
    """
    Ingest data from Kaggle dataset using kagglehub and load it into a pandas DataFrame.

    Returns:
        df (pandas.DataFrame): DataFrame containing the dataset.
    """
    print("Ingesting data...")
    # Download the dataset using kagglehub
    dataset_dir = kagglehub.dataset_download("mianbilal12/bitcoin-historical-data")
    
    # Find the first .csv file in the downloaded folder
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".csv"):
            csv_path = os.path.join(dataset_dir, filename)
            break
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    return df


# 2. Preprocess Data Task
@task
def preprocess_data(df):
    """
    Preprocess the data by creating additional features and splitting it into training, validation, and test sets.

    Args:
        df (pandas.DataFrame): Raw DataFrame containing the dataset.

    Returns:
        df_cleaned (pandas.DataFrame): Cleaned DataFrame ready for model training.
        X_train (pandas.DataFrame): Features for the training set.
        y_train (pandas.Series): Target for the training set.
        X_val (pandas.DataFrame): Features for the validation set.
        y_val (pandas.Series): Target for the validation set.
        X_test (pandas.DataFrame): Features for the test set.
        y_test (pandas.Series): Target for the test set.
    """
    print("Preprocessing data...")
    
    # Feature engineering: Creating lag and rolling features
    # Ensure the 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Adj_Close_Lag_1'] = df['Adj Close'].shift(1)
    df['Volume_Lag_1'] = df['Volume'].shift(1)
    df['A_30_day_avg_Adj_Close'] = df['Adj Close'].rolling(window=30).mean()
    df['A_30_day_volatility'] = df['Adj Close'].rolling(window=30).std()
    df['Daily_Return'] = df['Adj Close'].pct_change()
    df['Target'] = df['Adj Close'].shift(-1)

    # Extract date-related features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Week_of_Year'] = df['Date'].dt.isocalendar().week
    df['Day_of_Week'] = df['Date'].dt.weekday
    

    df_cleaned = df[['Volume_Lag_1', "Daily_Return","A_30_day_avg_Adj_Close", "A_30_day_volatility", "Year", "Month", "Day", "Target"]]
    df_cleaned = df_cleaned.dropna()
    # Define split proportions (70% training, 15% validation, 15% test)
    train_size = 0.7
    val_size = 0.15
    test_size = 0.15

    # Calculate split indices based on the dataset size
    num_samples = len(df_cleaned)
    train_end = int(train_size * num_samples)
    val_end = train_end + int(val_size * num_samples)

    # Split data into features (X) and target (y) for each dataset
    X_train = df_cleaned[:train_end].drop('Target', axis=1)
    y_train = df_cleaned[:train_end]['Target']
    X_val = df_cleaned[train_end:val_end].drop('Target', axis=1)
    y_val = df_cleaned[train_end:val_end]['Target']
    X_test = df_cleaned[val_end:].drop('Target', axis=1)
    y_test = df_cleaned[val_end:]['Target']

    return df_cleaned, X_train, y_train, X_val, y_val, X_test, y_test


# 3. Hyperparameter Tuning Task
@task
def hyperparameter_tuning(X_train, y_train):
    """
    Perform hyperparameter tuning for multiple models using GridSearchCV.

    Args:
        X_train (pandas.DataFrame): Features for the training set.
        y_train (pandas.Series): Target for the training set.

    Returns:
        best_models (dict): Dictionary containing the best models for each algorithm.
    """
    print("Performing hyperparameter tuning...")
    
    # Define models to tune
    models = {
        'RandomForest': RandomForestRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
    }

    # Define hyperparameter grids for each model
    param_grids = {
        'RandomForest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        'GradientBoosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.5],
            'max_depth': [3, 5, 7]
        }
        
    }

    # Perform grid search to find the best models
    best_models = {}
    
    for model_name, model in models.items():
        print(f"Tuning hyperparameters for {model_name}...")
        grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Store the best model after hyperparameter tuning
        best_models[model_name] = grid_search.best_estimator_
        print(f"Best hyperparameters for {model_name}: {grid_search.best_params_}")
    
    return best_models


# 4. Train Model Task
@task
def train_model(X_train, X_val, y_train, y_val, best_models):
    """
    Train models using the best hyperparameters and evaluate them on the validation set.

    Args:
        X_train (pandas.DataFrame): Features for the training set.
        X_val (pandas.DataFrame): Features for the validation set.
        y_train (pandas.Series): Target for the training set.
        y_val (pandas.Series): Target for the validation set.
        best_models (dict): Best models obtained from hyperparameter tuning.

    Returns:
        best_model (sklearn estimator): The model with the best R^2 score on the validation set.
    """
    print("Training models with best hyperparameters...")
    
    best_model = None
    best_r2 = -float("inf")

    # Train and evaluate each model
    for model_name, model in best_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        mse = mean_squared_error(y_val, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_val, y_pred)
        mspe = np.mean(np.square((y_val - y_pred) / y_val)) * 100  

        # Log metrics to MLflow
        with mlflow.start_run():
            mlflow.log_param('model_name', model_name)
            mlflow.log_metric('mse', mse)
            mlflow.log_metric('rmse', rmse)
            mlflow.log_metric('r2', r2)
            mlflow.log_metric('mspe', mspe)

        # Track the best model based on R2 score
        if mspe < 10:
            if r2 > best_r2:
                best_r2 = r2
                best_model = model

    return best_model


# 5. Register Model Task
@task
def register_model(best_model, X_val, y_val):
    """
    Register the best model based on MSPE (Mean Squared Percentage Error).

    Args:
        best_model (sklearn estimator): The model to be registered.
        X_val (pandas.DataFrame): Features for the validation set.
        y_val (pandas.Series): Target for the validation set.

    Returns:
        str: Status of model registration.
    """
    print("Registering the best model based on MSPE...")
    
    best_mspe = float("inf")

    # Predict and calculate MSPE for the best model
    y_pred = best_model.predict(X_val)
    mspe = np.mean(np.square((y_val - y_pred) / y_val)) * 100

    print(f"Best model MSPE: {mspe}")

    # Register the model if MSPE is better than the previous best
    if mspe < best_mspe:
        best_mspe = mspe
        
        # Log the model in MLflow
        with mlflow.start_run():
            mlflow.log_param("best_model", type(best_model).__name__)
            mlflow.log_metric("best_mspe", best_mspe)
            mlflow.sklearn.log_model(best_model, "best_model")
            mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/best_model", "BestModel")
        
        print("Best model registered successfully!")
        return "model_registered"
    else:
        print("No model found to register.")
        return "model_registration_failed"


# Define the main flow using Prefect
@flow
def main_flow():
    """
    Orchestrates the data ingestion, preprocessing, hyperparameter tuning, model training, 
    and model registration tasks.
    """
    # Ingest data
    df = ingest_data()

    # Preprocess data and split into training, validation, and test sets
    df_cleaned, X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(df)

    # Perform hyperparameter tuning to find the best models
    best_models = hyperparameter_tuning(X_train, y_train)

    # Train the best model based on the validation set
    best_model = train_model(X_train, X_val, y_train, y_val, best_models)

    # Register the best model
    register_model(best_model, X_val, y_val)


# Run the main flow if this script is executed directly
if __name__ == "__main__":
    main_flow()



