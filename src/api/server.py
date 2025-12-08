import uvicorn
from datetime import datetime
from typing import Union
from fastapi import FastAPI

app = FastAPI()

from pydantic import BaseModel

class PredictRequest(BaseModel):
    name: str
    birth_date: str
    gender: str
    education: str
    employeed: bool
    marital_status: str
    dependents: int
    property_area: str
    income: int
    coapplicant_income: int
    loan_amount: int
    loan_term: int
    credit_history: str

class PredictResponse(BaseModel):
    prediction: str
    



@app.get("/")
def read_root():
    return "this is the prediction api, start sending requests to /predict"

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running"}


@app.post("/predict")
def predict(data: PredictRequest):
    if data.birth_date:
        birth_date = datetime.strptime(data.birth_date, "%Y-%m-%d")
        age = datetime.now().year - birth_date.year
        if age < 18:
            return PredictResponse(prediction="rejected")
        else:
            if data.income/2 >  data.loan_amount/(data.loan_term/30):
                return PredictResponse(prediction="approved")
            else:
                return PredictResponse(prediction="rejected")
    else:
        return PredictResponse(prediction="rejected")


# uvicorn.run is handled by Docker CMD
# uvicorn.run(app, host="0.0.0.0", port=3003)