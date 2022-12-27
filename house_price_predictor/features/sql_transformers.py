from typing import Tuple

from pyspark.ml.feature import SQLTransformer


def date_converter() -> Tuple[str, SQLTransformer]:
    output_col = "converted_date"
    return output_col, SQLTransformer(
        statement=f"SELECT *, to_timestamp(date, \"yyyyMMdd'T'HHmmss\") AS {output_col} FROM __THIS__"
    )


def construction_age_creator() -> Tuple[str, SQLTransformer]:
    output_col = "construction_age"
    return output_col, SQLTransformer(
        statement=f"SELECT *, year(converted_date) - yr_built AS {output_col} FROM __THIS__"
    )


def renovation_age_creator() -> Tuple[str, SQLTransformer]:
    output_col = "renovation_age"
    return output_col, SQLTransformer(
        statement=f"SELECT *, year(converted_date) - greatest(yr_built, yr_renovated) AS {output_col} FROM __THIS__"
    )
