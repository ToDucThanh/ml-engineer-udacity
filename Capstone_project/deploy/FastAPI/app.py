import uvicorn
import logging 
import sys

from utils import model_fn, input_fn, predict_fn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

class Sample(BaseModel):
    Age: List[int]
    Sex: List[str]
    ChestPainType: List[str]
    RestingBP: List[float]
    Cholesterol: List[float]
    FastingBS: List[int]
    RestingECG: List[str]
    MaxHR: List[float]
    ExerciseAngina: List[str]
    Oldpeak: List[float]
    ST_Slope: List[str]

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    models["et"] = model_fn(model_dir="models")
    yield
    # Clean up the ML models and release the resources
    models.clear()

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
def predict(request: Sample):
    try:
        input_data = input_fn(request_body=request)
        output_data = predict_fn(data=input_data, et_clf=models["et"])
        status = "Successful"
    except Exception as e:
        logger.error(e)
        output_data = None
        status = "Fail"

    return {"class": output_data, "status": status}

if __name__ == "__main__":
    uvicorn.run(app="app:app", host="0.0.0.0", port=8000, reload=True)