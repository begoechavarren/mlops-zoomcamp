import pandas as pd

if "data_loader" not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if "test" not in globals():
    pass


@data_loader
def load_data_from_file(*args, **kwargs):
    filepath = "./mlops/homework_03/data/input/yellow_tripdata_2023-03.parquet"
    return pd.read_parquet(filepath)
