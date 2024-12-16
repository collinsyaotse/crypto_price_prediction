import pytest
import sys
import os
from unittest.mock import patch, MagicMock
from prefect import flow, task
import pandas as pd
# Add the 'model_management_scripts' directory to the system path
path_to_module = "C:\Users\Quavooo\Documents\crypto_price_prediction\model_management_scripts"

sys.path.append(path_to_module)

from model_management_scripts import main



@pytest.fixture
def mock_data():
    """Fixture to return mock data"""
    # Return a mock dataframe
    return pd.DataFrame({
        'Adj Close': [1, 2, 3, 4, 5],
        'Volume': [10, 20, 30, 40, 50],
        'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
    })

# Mocking the ingest function
@patch('main.ingest_fxn', return_value=None)
@patch('main.df', return_value=mock_data())
def test_ingest_data(mock_ingest, mock_df):
    result = main.ingest_data()
    assert result is not None
    assert isinstance(result, pd.DataFrame)
    mock_ingest.assert_called_once()

# Mocking the preprocess function
@patch('main.preprocess_data_fxn', return_value=(mock_data(), mock_data(), mock_data(), mock_data(), mock_data(), mock_data()))
def test_preprocess_data(mock_preprocess):
    df_cleaned, X_train, y_train, X_val, y_val, X_test, y_test = main.preprocess_data(mock_data())
    assert df_cleaned is not None
    assert len(X_train) > 0
    mock_preprocess.assert_called_once()

# Mocking the hyperparameter tuning function
@patch('main.hyperparameter_tuning_fxn', return_value={'RandomForest': 'best_model'})
def test_hyperparameter_tuning(mock_tuning):
    best_models = main.hyperparameter_tuning(mock_data(), mock_data())
    assert best_models == {'RandomForest': 'best_model'}
    mock_tuning.assert_called_once()

# Mocking the train model function
@patch('main.train_model_fxn', return_value='best_trained_model')
def test_train_model(mock_train):
    best_model = main.train_model(mock_data(), mock_data(), mock_data(), mock_data(), {'RandomForest': 'best_model'})
    assert best_model == 'best_trained_model'
    mock_train.assert_called_once()

# Mocking the register model function
@patch('main.register_model_fxn', return_value=None)
def test_register_model(mock_register):
    result = main.register_model('best_trained_model', mock_data(), mock_data())
    assert result is None
    mock_register.assert_called_once()

# Main flow test
@patch('main.ingest_data', return_value=mock_data())
@patch('main.preprocess_data', return_value=(mock_data(), mock_data(), mock_data(), mock_data(), mock_data(), mock_data()))
@patch('main.hyperparameter_tuning', return_value={'RandomForest': 'best_model'})
@patch('main.train_model', return_value='best_trained_model')
@patch('main.register_model', return_value=None)
def test_main_flow(mock_ingest, mock_preprocess, mock_tuning, mock_train, mock_register):
    main.main_flow()

    # Ensure the flow tasks were called
    mock_ingest.assert_called_once()
    mock_preprocess.assert_called_once()
    mock_tuning.assert_called_once()
    mock_train.assert_called_once()
    mock_register.assert_called_once()
