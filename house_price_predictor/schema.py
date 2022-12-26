from pyspark.sql.types import (
    StructType,
    StructField,
    LongType,
    StringType,
    DoubleType,
    IntegerType,
)

schema = StructType(
    [
        StructField("id", LongType(), True),
        StructField("date", StringType(), True),
        StructField("price", DoubleType(), True),
        StructField("bedrooms", DoubleType(), True),
        StructField("bathrooms", DoubleType(), True),
        StructField("sqft_living", DoubleType(), True),
        StructField("sqft_lot", DoubleType(), True),
        StructField("floors", DoubleType(), True),
        StructField("waterfront", IntegerType(), True),
        StructField("view", IntegerType(), True),
        StructField("condition", IntegerType(), True),
        StructField("grade", IntegerType(), True),
        StructField("sqft_above", DoubleType(), True),
        StructField("sqft_basement", DoubleType(), True),
        StructField("yr_built", IntegerType(), True),
        StructField("yr_renovated", IntegerType(), True),
        StructField("zipcode", IntegerType(), True),
        StructField("lat", DoubleType(), True),
        StructField("long", DoubleType(), True),
        StructField("sqft_living15", DoubleType(), True),
        StructField("sqft_lot15", DoubleType(), True),
    ]
)
