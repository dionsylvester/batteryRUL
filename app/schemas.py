from pydantic import BaseModel, Field
from typing import List

class TelemetryInput(BaseModel):
    discharge_time: float = Field(..., description="Duration required for the battery to fully deplete", example=6026.76)
    decrement_v: float = Field(..., description="Time interval for the battery voltage to drop from 3.6V to 3.4V", example=1220.00)
    max_v_discharge: float = Field(..., description="Highest voltage recorded while the battery is discharging", example=4.25)
    min_v_charge: float = Field(..., description="Lowest voltage recorded throughout the battery charging cycle", example=3.21)
    time_415v: float = Field(..., description="Time in which the battery voltage reaches 4.15V", example=5505.99)
    time_cc: float = Field(..., description="Period which the current continues steady at its maximum value", example=6161.00)
    charging_time: float = Field(..., description="Total time taken to fully charge the battery", example=10500.33)

class PredictionResponse(BaseModel):
    predicted_rul: float = Field(..., description="Estimated Remaining Useful Life (cycles)")
    status: str = Field(default="success")

class BatchTelemetryInput(BaseModel):
    batch: List[TelemetryInput]

class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    total_samples: int
    status: str = Field(default="success")