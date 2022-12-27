from collections import defaultdict
from datetime import datetime
from typing import List, Optional, Dict, Union

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

spark = SparkSession.builder.config("spark.driver.memory", "2g").getOrCreate()
sc = spark.sparkContext

pipeline_model = PipelineModel.load("models/best_model")


class Row(BaseModel):
    """The pydantic definition of each data row

    The values can all be string just because if not the validation would immediately break any request with invalid data.
    """

    id: int
    date: Optional[str]
    bedrooms: Optional[Union[float, str]]
    bathrooms: Optional[Union[float, str]]
    sqft_living: Optional[Union[float, str]]
    sqft_lot: Optional[Union[float, str]]
    floors: Optional[Union[float, str]]
    waterfront: Optional[Union[int, str]]
    view: Optional[Union[int, str]]
    condition: Optional[Union[int, str]]
    grade: Optional[Union[int, str]]
    sqft_above: Optional[Union[float, str]]
    sqft_basement: Optional[Union[float, str]]
    yr_built: Optional[Union[int, str]]
    yr_renovated: Optional[Union[int, str]]
    zipcode: Optional[Union[int, str]]
    lat: Optional[Union[float, str]]
    long: Optional[Union[float, str]]
    sqft_living15: Optional[Union[float, str]]
    sqft_lot15: Optional[Union[float, str]]

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
    warnings: List[str]


# I chose a PUT method because this is an idempotent method. POST definition is more for non-idempotent operations.
@app.put("/pred_price", response_model=List[RowPredictionResponse])
def predict(data: List[Row]) -> List[RowPredictionResponse]:
    warnings = _generate_warnings(data)

    # Performs the full feature engineering and computes the predictions
    df = spark.createDataFrame(data=map(vars, data), schema=schema, verifySchema=False)
    result_df = pipeline_model.transform(df).select("id", "prediction")

    # Joins the warnings with the predictions and returns them
    return [
        RowPredictionResponse(
            id=row.id, prediction=row.prediction, warnings=warnings[row.id]
        )
        for row in result_df.collect()
    ]


def _generate_warnings(data: List[Row]) -> Dict[int, List[str]]:
    """Checks if there are any missing/invalid features to add to the warnings"""
    warnings = defaultdict(list)
    for row in data:
        for row_property in row.schema()["properties"].keys():
            val = getattr(row, row_property)
            if row_property == "date":
                if _is_invalid_date(val):
                    warnings[row.id].append(
                        f"The 'date' format is invalid, so the date and all derived "
                        f"features will be imputed."
                    )
                    row.date = None
            elif row_property != "id" and _is_invalid_num(val):
                warnings[row.id].append(
                    f"The row property '{row_property}' is missing or invalid so "
                    f"it and its derived features will be imputed."
                )
    return warnings


def _is_invalid_num(num: str) -> bool:
    if num is None:
        return True
    try:
        float(num)
        return False
    except ValueError:
        return True


def _is_invalid_date(date: str) -> bool:
    if date is None:
        return True
    try:
        datetime.strptime(date, "%Y%m%dT%H%M%S")
        return False
    except ValueError:
        return True
