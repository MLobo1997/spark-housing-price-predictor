from copy import deepcopy
from typing import Dict

from pyspark.ml import Transformer, Estimator
from pyspark.ml.base import M
from pyspark.ml.util import MLWritable, MLReadable
from pyspark.sql import DataFrame
from pyspark.sql.functions import udf, col, avg, monotonically_increasing_id
from pyspark.sql.types import IntegerType


class ZipCodeRankerModel(Transformer, MLWritable, MLReadable):
    def __init__(self, price_rank_per_zipcode_dict: Dict[int, int]) -> None:
        super().__init__()
        self._price_rank_per_zipcode_dict = deepcopy(price_rank_per_zipcode_dict)
        self.zipcode_rank_udf = udf(
            lambda x: self._price_rank_per_zipcode_dict.get(x), IntegerType()
        )

    def _transform(self, dataset: DataFrame) -> DataFrame:
        return dataset.withColumn(
            "zipcode_price_rank", self.zipcode_rank_udf(col("zipcode"))
        )


class ZipcodeRanker(Estimator):
    def _fit(self, dataset: DataFrame) -> M:
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
