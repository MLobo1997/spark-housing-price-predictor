from typing import List, Union

from pyspark.ml import Estimator, Pipeline, Transformer
from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler, PCA
from pyspark.sql import DataFrame
from pyspark.sql.types import IntegerType, LongType, FloatType, DoubleType

from house_price_predictor.features.sql_transformers import (
    date_converter,
    construction_age_creator,
    renovation_age_creator,
)
from house_price_predictor.features.zipcode_ranker import ZipcodeRanker

FEATS_TO_EXCLUDE = {"price", "id", "yr_built", "yr_renovated"}

FEATURE_DTYPES = [
    IntegerType.typeName(),
    LongType.typeName(),
    FloatType.typeName(),
    DoubleType.typeName(),
]


def get_all_feat_names(dataset: DataFrame, feats_to_exclude=None) -> List[str]:
    """Gets all features of a Dataframe that are numerical and should not be excluded"""
    if feats_to_exclude is None:
        feats_to_exclude = FEATS_TO_EXCLUDE

    original_fields = [
        field.name
        for field in dataset.schema.fields
        if (
            (field.dataType.typeName() in FEATURE_DTYPES)
            and (field.name not in feats_to_exclude)
        )
    ]

    return original_fields


def create_fitted_pipeline(*, dataset: DataFrame, model: Estimator = None):
    """Creates a pipeline with all the feature engineering transformers and fits it"""
    all_feat_to_include_names = get_all_feat_names(dataset)

    _, dateconverter = date_converter()
    # Variable where all transformers will be
    stages: List[Union[Estimator, Transformer]] = [ZipcodeRanker(), dateconverter]

    # Add all SQL Transformers. Done separately so that columns they create are added to `all_feat_to_include_names`
    for feat_name, transformer in [
        construction_age_creator(),
        renovation_age_creator(),
    ]:
        all_feat_to_include_names.append(feat_name)
        stages.append(transformer)

    final_feat_col = "features"
    stages += [
        Imputer(
            inputCols=all_feat_to_include_names,
            outputCols=all_feat_to_include_names,
            strategy="mean",
        ),
        VectorAssembler(
            inputCols=all_feat_to_include_names,
            outputCol="vector",
            handleInvalid="keep",
        ),
        StandardScaler(
            inputCol="vector", outputCol="scaled", withMean=True, withStd=True
        ),
        PCA(k=13, inputCol="scaled", outputCol=final_feat_col),
    ]

    if model:
        stages.append(model)
    pipeline = Pipeline(stages=stages)

    return pipeline.fit(dataset)
