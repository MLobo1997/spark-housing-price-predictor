from pyspark.sql import DataFrame
from pyspark.sql.functions import col, avg, abs


def evaluate_relative_error(dataset: DataFrame) -> float:
    """Computes the average relative error in a dataframe with predictions."""
    return (
        dataset.select(
            abs((col("price") - col("prediction")) / col("price")).alias(
                "relative_error"
            )
        )
        .select(avg("relative_error"))
        .toPandas()
        .iloc[0, 0]
    )
