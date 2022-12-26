import json
from typing import Dict

from pyspark.ml import Transformer, Estimator
from pyspark.ml.util import MLWritable, MLReadable, MLReader, RL, MLWriter
from pyspark.sql import DataFrame
from pyspark.sql.functions import udf, col, avg, monotonically_increasing_id
from pyspark.sql.types import IntegerType


class ZipCodeRankerModelWriter(MLWriter):
    def __init__(self, instance: "ZipCodeRankerModel"):
        super().__init__()
        self.instance = instance

    def saveImpl(self, path: str) -> None:
        with open(path, "w") as file:
            file.write(json.dumps(self.instance.price_rank_per_zipcode_dict))


class ZipCodeRankerModelReader(MLReader):
    def load(self, path: str) -> RL:
        with open(path, "r") as file:
            price_rank_per_zipcode_dict = json.load(file)
        return ZipCodeRankerModel(
            {int(k): int(v) for k, v in price_rank_per_zipcode_dict.items()}
        )


class ZipCodeRankerModel(Transformer, MLWritable, MLReadable["ZipCodeRankerModel"]):
    def write(self) -> MLWriter:
        return ZipCodeRankerModelWriter(self)

    @classmethod
    def read(cls) -> MLReader[RL]:
        return ZipCodeRankerModelReader()

    def __init__(self, price_rank_per_zipcode_dict: Dict[int, int]) -> None:
        super().__init__()
        self.price_rank_per_zipcode_dict = price_rank_per_zipcode_dict

    def _transform(self, dataset: DataFrame) -> DataFrame:
        zipcode_rank_udf = udf(
            lambda x: self.price_rank_per_zipcode_dict.get(x), IntegerType()
        )
        return dataset.withColumn(
            "zipcode_price_rank", zipcode_rank_udf(col("zipcode"))
        )


class ZipcodeRanker(Estimator):
    def _fit(self, dataset: DataFrame) -> Transformer:
        zipcodes_price_df = (
            dataset.groupby("zipcode")
            .agg(avg("price").alias("price_avg"))
            .sort("price_avg", ascending=False)
            .drop("price_avg")
            .withColumn("zipcode_price_rank", monotonically_increasing_id())
        )
        return ZipCodeRankerModel(
            {
                row["zipcode"]: row["zipcode_price_rank"]
                for row in zipcodes_price_df.collect()
            }
        )
