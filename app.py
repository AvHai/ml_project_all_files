from fastapi import FastAPI
from pydantic import BaseModel
from src.pipeline.predict_pipeline import predict_flight_delay, CustomException

app = FastAPI(title="Flight Delay Prediction API", version="1.0")

class FlightRequest(BaseModel):
    year: int
    month: int
    day: int
    airline: str
    origin_airport: str
    destination_airport: str
    scheduled_departure: int
    scheduled_time: int = None
    distance: float = None

@app.post("/predict")
def predict_flight(data: FlightRequest):
    result = predict_flight_delay(
        year=data.year,
        month=data.month,
        day=data.day,
        airline=data.airline,
        origin_airport=data.origin_airport,
        destination_airport=data.destination_airport,
        scheduled_departure=data.scheduled_departure,
        scheduled_time=data.scheduled_time,
        distance=data.distance
    )
    return {"status": "success", "prediction": result}
