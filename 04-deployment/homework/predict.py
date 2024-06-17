import pickle

import numpy as np
import pandas as pd
import typer

app = typer.Typer()

CATEGORICAL_COLS = ["PULocationID", "DOLocationID"]


def read_data(year, month):
    filename = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"

    df = pd.read_parquet(filename)

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].fillna(-1).astype("int").astype("str")
    df["ride_id"] = f"{year}/{month:02d}_" + df.index.astype("str")

    return df


def run_model(df: pd.DataFrame, dv, model) -> pd.DataFrame:
    dicts = df[CATEGORICAL_COLS].to_dict(orient="records")
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    df_result = df[["ride_id"]].copy()
    df_result["prediction"] = y_pred
    return df_result


def save_output(df, output_file) -> None:
    df.to_parquet(
        output_file,
        engine="pyarrow",
        compression=None,
        index=False,
    )


@app.command()
def predict(
    year: int = typer.Option(..., help="The year of the trip data"),
    month: int = typer.Option(..., help="The month of the trip data"),
    model_file: str = typer.Option(..., help="Path to the model file"),
    output_file: str = typer.Option(..., help="Path to the output file"),
) -> None:
    with open(model_file, "rb") as f_in:
        dv, model = pickle.load(f_in)

    df = read_data(year, month)
    df_result = run_model(df, dv, model)
    save_output(df_result, output_file)

    typer.echo(f"Mean predicted duration: {np.mean(df_result['prediction'])}")


if __name__ == "__main__":
    app()
