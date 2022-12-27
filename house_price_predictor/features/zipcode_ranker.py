import json
from typing import Dict

from pyspark.ml import Transformer, Estimator
from pyspark.ml.util import (
    MLWritable,
    MLReadable,
    MLReader,
    RL,
    MLWriter,
    DefaultParamsWriter,
)
from pyspark.sql import DataFrame
from pyspark.sql.functions import udf, col, avg, monotonically_increasing_id
from pyspark.sql.types import IntegerType

PERSISTENCE_FILE = "price_ranks.json"


class ZipcodeRanker(Estimator):
    """Computes the ranking of zipcodes by average price."""

    def _fit(self, dataset: DataFrame) -> Transformer:
        zipcodes_price_df = (
            dataset.groupby("zipcode")
            .agg(avg("price").alias("price_avg"))
            .sort("price_avg", ascending=False)
            .withColumn("zipcode_price_rank", monotonically_increasing_id())
            .drop("price_avg")
        )
        return ZipCodeRankerModel(
            {
                row["zipcode"]: row["zipcode_price_rank"]
                for row in zipcodes_price_df.collect()
            }
        )


class ZipCodeRankerModel(Transformer, MLWritable, MLReadable["ZipCodeRankerModel"]):
    """Creates a new column with the zipcode position within the ranking."""

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

    def write(self) -> MLWriter:
        return ZipCodeRankerModelWriter(self)

    @classmethod
    def read(cls) -> MLReader[RL]:
        return ZipCodeRankerModelReader()


class ZipCodeRankerModelWriter(MLWriter):
    """Necessary to make ZipcodeRankerModel serializable."""

    def __init__(self, instance: "ZipCodeRankerModel"):
        super().__init__()
        self.instance = instance

    def saveImpl(self, path: str) -> None:
        jsonParams = {"language": "Python"}
        DefaultParamsWriter.saveMetadata(
            self.instance, path, self.sc, paramMap=jsonParams
        )
        with open(f"{path}/{PERSISTENCE_FILE}", "w") as file:
            file.write(json.dumps(self.instance.price_rank_per_zipcode_dict))


class ZipCodeRankerModelReader(MLReader):
    """Necessary to make ZipcodeRankerModel serializable."""

    def load(self, path: str) -> RL:
        with open(f"{path}/{PERSISTENCE_FILE}", "r") as file:
            price_rank_per_zipcode_dict = json.load(file)
        return ZipCodeRankerModel(
            {int(k): int(v) for k, v in price_rank_per_zipcode_dict.items()}
        )
