from collections import defaultdict
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

from house_price_predictor.schema import schema

app = FastAPI(
    title="House Price Predictor",
    description="An endpoint to estimate the value of your house.",
    version="0.0.1",
)

spark = SparkSession.builder.config("spark.driver.memory", "4g").getOrCreate()
sc = spark.sparkContext

pipeline_model = PipelineModel.load("models/lr-1672019228")


class Row(BaseModel):
    """The pydantic definition of each data row"""

    id: int
    date: Optional[str]
    bedrooms: Optional[float]
    bathrooms: Optional[float]
    sqft_living: Optional[float]
    sqft_lot: Optional[float]
    floors: Optional[float]
    waterfront: Optional[int]
    view: Optional[int]
    condition: Optional[int]
    grade: Optional[int]
    sqft_above: Optional[float]
    sqft_basement: Optional[float]
    yr_built: Optional[int]
    yr_renovated: Optional[int]
    zipcode: Optional[int]
    lat: Optional[float]
    long: Optional[float]
    sqft_living15: Optional[float]
    sqft_lot15: Optional[float]

    class Config:
        """An example to make the openapi definition look prettier."""

        schema_extra = {
            "example": {
                "id": 6414100192,
                "date": "20141209T000000",
                "bedrooms": 3.0,
                "bathrooms": 2.25,
                "sqft_living": 2570.0,
                "sqft_lot": 7242.0,
                "floors": 2.0,
                "waterfront": 0,
                "view": 0,
                "condition": 3,
                "grade": 7,
                "sqft_above": 2170.0,
                "sqft_basement": 400.0,
                "yr_built": 1951,
                "yr_renovated": 1991,
                "zipcode": 98125,
                "lat": 47.721,
                "long": -122.319,
                "sqft_living15": 1690.0,
                "sqft_lot15": 7639.0,
            },
        }


# using IDs is a good API design pattern for ML
class RowPredictionResponse(BaseModel):
    """The pydantic definition of the response for each row."""

    id: int
    prediction: float
    warnings: Optional[List[str]]


def is_invalid_date(date: str) -> bool:
    try:
        datetime.strptime(date, "%Y%m%dT%H%M%S")
        return False
    except ValueError:
        return True


# I chose a PUT method because this is an idempotent method. POST definition is more for non-idempotent operations.
@app.put("/pred_price", response_model=List[RowPredictionResponse])
def predict(data: List[Row]) -> List[RowPredictionResponse]:
    # Checks if there are any missing features to add to the warnings
    warnings = defaultdict(list)
    for row in data:
        for row_property in row.schema()["properties"].keys():
            if getattr(row, row_property) is None and row_property != "id":
                warnings[row.id].append(
                    f"The row property '{row_property}' is missing so it and its derived features will be imputed."
                )
            elif row_property == "date" and is_invalid_date(getattr(row, row_property)):
                warnings[row.id].append(
                    f"The 'date' format is invalid, so the date and all derived features will be imputed."
                )
                row.date = None

    # Performs the full feature engineering and computes the predictions
    df = spark.createDataFrame(data=map(vars, data), schema=schema)
    result_df = pipeline_model.transform(df).select("id", "prediction")

    # Joins the warnings with the predictions and returns them
    return [
        RowPredictionResponse(
            id=row.id, prediction=row.prediction, warnings=warnings[row.id]
        )
        for row in result_df.collect()
    ]
