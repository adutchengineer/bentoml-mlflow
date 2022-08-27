from __future__ import annotations

import json
import numpy as np
import bentoml
import pandas as pd
import pydantic
from bentoml.io import JSON, PandasSeries, PandasDataFrame


class PreProcessor(bentoml.Runnable):
    SUPPORTED_RESOURCES = ()
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        pass

    @bentoml.Runnable.method(batchable=True)
    def remove_na(self, df: pd.DataFrame):
        return df.dropna()


preprocessor_runner = bentoml.Runner(PreProcessor)
runner = bentoml.mlflow.get('sklearn_house_data').to_runner()
svc = bentoml.Service('sklearn_house_data', runners=[
                      preprocessor_runner, runner])


class File(pydantic.BaseModel):
    path: str


file_input = JSON(
    pydantic_model=File,
    validate_json=True)


@svc.api(
    input=file_input,
    output=JSON(),
    route='v1/file/'
)
def predictions(file_input: File) -> json:
    file_input = file_input.path
    houses = pd.read_csv(file_input)
    df = preprocessor_runner.remove_na.run(houses)
    prices = runner.run(df).flatten()
    return {'prices': prices}


@svc.api(
    input=PandasDataFrame(),
    output=JSON(),
    route='v1/predict/'
)
def predict(house_df: pd.DataFrame) -> json:
    house_df.columns = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
                        'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']
    prices = runner.run(house_df.astype(float))
    return {'price': prices}
