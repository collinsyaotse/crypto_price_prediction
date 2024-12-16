import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

# FastAPI app initialization
app = FastAPI()

# Path to the model file
model_path = r"C:\Users\Quavooo\Documents\crypto_price_prediction\model_management_scripts\mlruns\models\BestModel\model.pkl"

# Load the model from the pickle file at the start of the FastAPI app
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Define the input data schema using Pydantic
class PredictionInput(BaseModel):
    Volume_Lag_1: float
    Daily_Return: float
    A_30_day_avg_Adj_Close: float
    A_30_day_volatility: float
    Year: int
    Month: int
    Day: int

    
class PredictionOutput(BaseModel):
    prediction: float

# Define the API endpoint to make predictions
@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    # Convert the input data to the format expected by the model (as a numpy array)
    input_array = np.array([[input_data.Volume_Lag_1, input_data.Daily_Return,
                             input_data.A_30_day_avg_Adj_Close, input_data.A_30_day_volatility,
                             input_data.Year, input_data.Month, input_data.Day]])

    # Make a prediction using the loaded model
    try:
        prediction = model.predict(input_array)
        return PredictionOutput(prediction=float(prediction[0]))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

# Add a simple root path to avoid the 404 error
@app.get("/")
def read_root():
    return {"message": "Welcome to the Crypto Price Prediction API!"}

# Run the FastAPI app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
