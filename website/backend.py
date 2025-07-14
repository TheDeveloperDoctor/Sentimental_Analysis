from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import pickle
from pydantic import BaseModel
import pandas as pd
from typing import Literal
import uvicorn

app = FastAPI()
current_dir = Path(__file__).parent

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def prediction_to_string(prediction):
    switch = {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    }
    return switch.get(prediction, "Unknown")

# Load model and preprocessor
model = None
preprocessor = None

try:
    model_path = current_dir / "RandomForest.pkl"
    preprocessor_path = current_dir / "data_preprocessor.pkl"
    
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(preprocessor_path, 'rb') as preprocessor_file:
        preprocessor = pickle.load(preprocessor_file)
    print("Model and preprocessor loaded successfully!")
except Exception as e:
    print(f"Error loading model or preprocessor: {str(e)}")

class TextInput(BaseModel):
    text: str
    age_group: Literal['18-24', '25-34', '35-44', '45-54', '55+'] = '25-34'
    time_of_tweet: Literal['Morning', 'Afternoon', 'Evening', 'Night'] = 'Afternoon'

@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open(current_dir / "index.html", "r") as file:
            content = file.read()
        return HTMLResponse(content=content)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read index.html: {str(e)}"
        )
        
@app.post("/predict")
async def predict_sentiment(text_input: TextInput):
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=500,
            detail="Model or preprocessor not loaded."
        )

    try:
        # Create DataFrame with correct column names and values
        input_df = pd.DataFrame({
            'clean_text': [text_input.text],
            'age_group': [text_input.age_group],  # Use the actual input value
            'time_of_tweet': [text_input.time_of_tweet]  # Use the actual input value
        })
        
        # Preprocess the input data
        processed_data = preprocessor.transform(input_df)
        
        # Make prediction
        prediction = model.predict(processed_data)
        
        # If your model supports probabilities
        try:
            probabilities = model.predict_proba(processed_data)
            confidence = probabilities.max()
        except AttributeError:
            confidence = None
        
        return {
            "prediction": prediction_to_string(prediction[0]),
            "confidence": float(confidence) if confidence is not None else None
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )

if __name__ == "__main__":
    print("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)