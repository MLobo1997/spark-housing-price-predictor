from typing import List

from fastapi import FastAPI
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from house_price_predictor.schema import schema

app = FastAPI()

spark = SparkSession.builder.config("spark.driver.memory", "4g").getOrCreate()
sc = spark.sparkContext

pipeline_model = PipelineModel.load("models/lr-1672019228")

from pydantic import BaseModel

class Row(BaseModel):
    id: int
    date: str
    price: float
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: float
    sqft_basement: float
    yr_built: int
    yr_renovated: int
    zipcode: int
    lat: float
    long: float
    sqft_living15: float
    sqft_lot15: float


class Data(BaseModel):
    rows: List[Row]

# PUT because this is an idempotent method and POST definition is more for non idempoen
@app.put("/pred_price")
def pred(data: List[dict]):
    # MAY LIST?
    df = spark.createDataFrame(data=data, schema=schema)
    # using IDs is a good API design pattern for ML
    result_df = pipeline_model.transform(df).select("id", "prediction")
    return list(map(lambda x: x.asDict(), result_df.collect()))
