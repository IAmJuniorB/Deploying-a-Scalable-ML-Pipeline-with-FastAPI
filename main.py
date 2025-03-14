import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from ml.data import apply_label, process_data
from ml.model import inference, load_model

# Define the path for the saved model and encoder
model_path = os.path.join(os.getcwd(), "model", "model.pkl")
encoder_path = os.path.join(os.getcwd(), "model", "encoder.pkl")

# Load the model and encoder
model = load_model(model_path)
encoder = load_model(encoder_path)

# Define the Pydantic model for input data
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(..., example="Married-civ-spouse", alias="marital-status")
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

# Create the FastAPI app
app = FastAPI()

# Create a GET request on the root
@app.get("/")
async def get_root():
    """Return a welcome message."""
    return {"message": "Hello from the API!"}

# Create a POST request for model inference
@app.post("/data/")
async def post_inference(data: Data):
    """Make model inference based on input data."""
    # Convert the Pydantic model into a dict
    data_dict = data.dict()
    
    # Clean up the dict to turn it into a Pandas DataFrame
    data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data = pd.DataFrame.from_dict(data)
    
    # Define categorical features
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    
    # Process the data
    data_processed, _, _, _ = process_data(
        data, categorical_features=cat_features, label=None, training=False, encoder=encoder
    )
    
    # Make model inference
    _inference = inference(model, data_processed)
    
    # Apply label
    result = apply_label(_inference)
    
    return {"result": result}