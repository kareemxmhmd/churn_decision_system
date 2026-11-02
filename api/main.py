from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.inference import load_inference_artifacts, predict_churn

ml_artifacts: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
     model, threshold = load_inference_artifacts()
     ml_artifacts["model"] = model
     ml_artifacts["threshold"] = threshold
     yield
     ml_artifacts.clear()


app = FastAPI(
          title="Customer Churn Prediction API",
          description="Calibrated XGBoost churn prediction service with cost-aware threshold.",
          version="1.0.0",
          lifespan=lifespan
)


class CustomerFeatures(BaseModel):
     Tenure: Optional[float] = Field(default=None, ge=0, description="Tenure in months")
     WarehouseToHome: Optional[float] = Field(default=None, ge=0, description="Distance from warehouse to home")
     NumberOfDeviceRegistered: int = Field(..., ge=1, description="Number of registered devices")
     SatisfactionScore: int = Field(..., ge=1, le=5, description="Customer satisfaction score (1-5)")
     NumberOfAddress: int = Field(..., ge=1, description="Number of addresses on file")
     DaySinceLastOrder: float = Field(..., ge=0, description="Days since last order")
     CashbackAmount: float = Field(..., ge=0, description="Cashback amount received")
     PreferedOrderCat: str = Field(..., description="Preferred order category")
     MaritalStatus: str = Field(..., description="Marital status (Single, Married, Divorced)")
     Complain: int = Field(..., ge=0, le=1, description="1 if customer complained in last month, else 0")

     model_config = ConfigDict(
          json_schema_extra={
               "example": {
                    "Tenure": 2.0,
                    "WarehouseToHome": 15.0,
                    "NumberOfDeviceRegistered": 4,
                    "SatisfactionScore": 3,
                    "NumberOfAddress": 3,
                    "DaySinceLastOrder": 1.0,
                    "CashbackAmount": 145.0,
                    "PreferedOrderCat": "Mobile Phone",
                    "MaritalStatus": "Single",
                    "Complain": 1,
               }
          }
     )


class PredictionResponse(BaseModel):
     churn_probability: float
     churn_prediction: int
     threshold_used: float
     recommended_action: str


@app.get("/health")
def health_check() -> dict[str, Any]:
     if "model" not in ml_artifacts:
          raise HTTPException(status_code=503, detail="Model artifacts are not loaded.")
     return {
          "status" : "healthy",
          "threshold": round(float(ml_artifacts)["threshold"], 4),
     }

     
@app.post("/predict", response_model=PredictionResponse)
def predict_single(customer: CustomerFeatures) -> PredictionResponse:
    try:
        df_res = predict_churn(
            data=customer.model_dump(),
            model=ml_artifacts.get("model"),
            threshold=ml_artifacts.get("threshold"),
        )
        row = df_res.iloc[0]
        pred = int(row["churn_prediction"])
        return PredictionResponse(
            churn_probability=float(row["churn_probability"]),
            churn_prediction=pred,
            threshold_used=float(row["threshold_used"]),
            recommended_action="Send retention incentive" if pred == 1 else "No action needed",
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(customers: list[CustomerFeatures]) -> list[PredictionResponse]:
    if not customers:
        raise HTTPException(status_code=400, detail="Customer list cannot be empty.")
    try:
        records = [c.model_dump() for c in customers]
        df_res = predict_churn(
            data=records,
            model=ml_artifacts.get("model"),
            threshold=ml_artifacts.get("threshold"),
        )
        return [
            PredictionResponse(
                churn_probability=float(row["churn_probability"]),
                churn_prediction=int(row["churn_prediction"]),
                threshold_used=float(row["threshold_used"]),
                recommended_action=(
                    "Send retention incentive"
                    if int(row["churn_prediction"]) == 1
                    else "No action needed"
                ),
            )
            for _, row in df_res.iterrows()
        ]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc