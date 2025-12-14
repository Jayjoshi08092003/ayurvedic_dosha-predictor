import pandas as pd
from catboost import CatBoostClassifier
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os
# IMPORT THE CORS MIDDLEWARE FIX
from fastapi.middleware.cors import CORSMiddleware 

# --- 1. Configuration & Feature List ---
MODEL_PATH = 'catboost_prakriti_model.cbm' 

# The EXACT list of 29 features your model was trained on, IN ORDER:
FEATURE_COLUMNS = [
    "Body Size", "Body Weight", "Height", "Bone Structure", "Complexion",
    "General feel of skin", "Texture of Skin", "Hair Color", "Appearance of Hair",
    "Shape of face", "Eyes", "Eyelashes", "Blinking of Eyes", "Cheeks", "Nose",
    "Teeth and gums", "Lips", "Nails", "Appetite", "Liking tastes",
    "Metabolism Type", "Climate Preference", "Stress Levels", "Sleep Patterns",
    "Dietary Habits", "Physical Activity Level", "Water Intake", "Digestion Quality",
    "Skin Sensitivity"
]

# --- 2. Load the CatBoost Model ---
try:
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    print(f"✅ Successfully loaded CatBoost model from {MODEL_PATH}")
except Exception as e:
    print(f"FATAL ERROR: Could not load model from {MODEL_PATH}. Check file existence.")
    raise

# --- 3. Define Input Schema (Pydantic) ---
class DoshaInput(BaseModel):
    # Dynamically define Pydantic fields based on FEATURE_COLUMNS, converting to snake_case
    __annotations__ = {col.lower().replace(' ', '_'): str for col in FEATURE_COLUMNS}
    
    body_size: str = "Medium"
    height: str = "Average"
    metabolism_type: str = "fast"
    
    def to_pandas_row(self):
        data = self.dict()
        row = {}
        for original_col in FEATURE_COLUMNS:
            snake_case_col = original_col.lower().replace(' ', '_')
            if snake_case_col in data:
                row[original_col] = data[snake_case_col]
        
        return pd.DataFrame([row], columns=FEATURE_COLUMNS)

# --- 4. Initialize FastAPI App ---
app = FastAPI(
    title="Dosha Prediction API",
    description="Endpoint for predicting Ayurvedic Dosha using a trained CatBoost model."
)

# --- VITAL FIX: CORS Configuration ---
# This middleware allows your local HTML file (the client) to send requests 
# to the FastAPI server running on a different port/origin, resolving the 405 error.
origins = ["*"] # Use "*" for local development to allow any origin

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],    # Allows OPTIONS (pre-flight) and POST
    allow_headers=["*"],
)
# ------------------------------------

# --- 5. Define the Prediction Endpoint ---
@app.post("/predict_dosha")
def predict_dosha(data: DoshaInput):
    """
    Accepts a JSON body with 29 categorical features and returns the predicted Dosha.
    """
    try:
        input_df = data.to_pandas_row()
        
        prediction = model.predict(input_df)
        predicted_dosha = prediction[0][0]

        return {
            "predicted_dosha": predicted_dosha,
            "status": "success",
            "model_path": MODEL_PATH
        }

    except Exception as e:
        return {
            "predicted_dosha": None,
            "status": "error",
            "message": f"Prediction failed due to internal server error: {str(e)}"
        }

# --- 6. Run the Server (Local Testing) ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)