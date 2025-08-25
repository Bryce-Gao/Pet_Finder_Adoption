from typing import Optional, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
from google.cloud import bigquery
import os

# Config from env (defaults are set for your project/model)
PROJECT_ID = os.getenv("PROJECT_ID", "msds434-final-project-466304")
DATASET = os.getenv("BQ_DATASET", "petfinder_data")
MODEL = os.getenv("BQ_MODEL", "auto_model_1")

MODEL_FQN = f"`{PROJECT_ID}.{DATASET}.{MODEL}`"

app = FastAPI(title="PetFinder AutoML Microservice", version="1.0")

class Features(BaseModel):
    # Numbers (INT64/FLOAT64)
    Age: Optional[int] = None
    Breed1: Optional[int] = None
    Breed2: Optional[int] = None
    Color1: Optional[int] = None
    Color2: Optional[int] = None
    Color3: Optional[int] = None
    Dewormed: Optional[int] = None
    Fee: Optional[int] = None
    FurLength: Optional[int] = None
    Gender: Optional[int] = None
    Health: Optional[int] = None
    MaturitySize: Optional[int] = None
    PhotoAmt: Optional[float] = None
    Quantity: Optional[int] = None
    State: Optional[int] = None
    Sterilized: Optional[int] = None
    Type: Optional[int] = None
    Vaccinated: Optional[int] = None
    VideoAmt: Optional[int] = None
    # Strings
    Name: Optional[str] = None
    RescuerID: Optional[str] = None

def build_query_and_params(f: Features):
    """
    Build a parameterized ML.PREDICT query against BigQuery ML AutoML model.
    We select a single-row table made from parameters to feed into ML.PREDICT.
    """
    # Column order matches your feature schema
    cols = [
        "Age","Breed1","Breed2","Color1","Color2","Color3","Dewormed","Fee",
        "FurLength","Gender","Health","MaturitySize","Name","PhotoAmt","Quantity",
        "RescuerID","State","Sterilized","Type","Vaccinated","VideoAmt"
    ]

    # Build SELECT list like: @Age AS Age, @Breed1 AS Breed1, ...
    select_exprs = []
    params = []

    # Helper to append one param
    def add_param(name: str, value: Any, bq_type: str):
        select_exprs.append(f"@{name} AS {name}")
        params.append(bigquery.ScalarQueryParameter(name, bq_type, value))

    # Map pydantic types -> BigQuery parameter types
    add_param("Age", f.Age, "INT64")
    add_param("Breed1", f.Breed1, "INT64")
    add_param("Breed2", f.Breed2, "INT64")
    add_param("Color1", f.Color1, "INT64")
    add_param("Color2", f.Color2, "INT64")
    add_param("Color3", f.Color3, "INT64")
    add_param("Dewormed", f.Dewormed, "INT64")
    add_param("Fee", f.Fee, "INT64")
    add_param("FurLength", f.FurLength, "INT64")
    add_param("Gender", f.Gender, "INT64")
    add_param("Health", f.Health, "INT64")
    add_param("MaturitySize", f.MaturitySize, "INT64")
    add_param("Name", f.Name, "STRING")
    add_param("PhotoAmt", f.PhotoAmt, "FLOAT64")
    add_param("Quantity", f.Quantity, "INT64")
    add_param("RescuerID", f.RescuerID, "STRING")
    add_param("State", f.State, "INT64")
    add_param("Sterilized", f.Sterilized, "INT64")
    add_param("Type", f.Type, "INT64")
    add_param("Vaccinated", f.Vaccinated, "INT64")
    add_param("VideoAmt", f.VideoAmt, "INT64")

    select_clause = ", ".join(select_exprs)
    query = f"""
    SELECT * FROM ML.PREDICT(
      MODEL {MODEL_FQN},
      (SELECT {select_clause})
    )
    """
    return query, params

@app.get("/")
def health():
    return {"status": "ok", "model": f"{PROJECT_ID}.{DATASET}.{MODEL}"}

@app.post("/predict")
def predict(features: Features):
    client = bigquery.Client(project=PROJECT_ID)
    query, params = build_query_and_params(features)
    job_config = bigquery.QueryJobConfig(query_parameters=params)
    job = client.query(query, job_config=job_config)
    rows = list(job.result())
    if not rows:
        return {"error": "No prediction returned"}
    # Return the entire row as JSON to avoid guessing predicted column names
    result: Dict[str, Any] = dict(rows[0].items())
    return {"prediction": result}
