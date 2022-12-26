from collections import defaultdict
from datetime import datetime
from typing import List, Optional, Any

from fastapi import FastAPI
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

from house_price_predictor.schema import schema

app = FastAPI()

spark = SparkSession.builder.config("spark.driver.memory", "4g").getOrCreate()
sc = spark.sparkContext

pipeline_model = PipelineModel.load("models/lr-1672019228")

from pydantic import BaseModel, validator


class Row(BaseModel):
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


# using IDs is a good API design pattern for ML
class RowPredictionResponse(BaseModel):
    id: int
    prediction: float
    warnings: Optional[List[str]]


def invalid_date(date: str) -> bool:
    try:
        datetime.strptime(date, "%Y%m%dT%H%M%S")
        return False
    except ValueError:
        return True


# PUT because this is an idempotent method and POST definition is more for non idempoen
@app.put("/pred_price", response_model=List[RowPredictionResponse])
def predict(data: List[Row]) -> List[RowPredictionResponse]:
    warnings = defaultdict(list)
    for row in data:
        for row_property in row.schema()["properties"].keys():
            if getattr(row, row_property) is None and row_property != "id":
                warnings[row.id].append(
                    f"The row property '{row_property}' is missing so it and its derived features will be imputed."
                )
            elif row_property == "date" and invalid_date(getattr(row, row_property)):
                warnings[row.id].append(
                    f"The 'date' format is invalid, so the date and all derived features will be imputed."
                )
                row.date = None

    df = spark.createDataFrame(data=map(vars, data), schema=schema)
    result_df = pipeline_model.transform(df).select("id", "prediction")
    return [
        RowPredictionResponse(
            id=row.id, prediction=row.prediction, warnings=warnings[row.id]
        )
        for row in result_df.collect()
    ]
