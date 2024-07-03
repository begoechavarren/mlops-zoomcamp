#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import sys

import pandas as pd


def get_input_path(
    year: int,
    month: int,
) -> str:
    default_input_pattern = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    input_pattern = os.getenv("INPUT_FILE_PATTERN", default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(
    year: int,
    month: int,
) -> str:
    default_output_pattern = "s3://nyc-duration-prediction-alexey/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet"
    output_pattern = os.getenv("OUTPUT_FILE_PATTERN", default_output_pattern)
    return output_pattern.format(year=year, month=month)


def prepare_data(
    df: pd.DataFrame,
    year: int,
    month: int,
    categorical: list[str],
) -> pd.DataFrame:
    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")
    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")
    return df


def read_data(
    year: int,
    month: int,
    filename: str,
) -> pd.DataFrame:
    s3_endpoint_url = os.getenv("S3_ENDPOINT_URL")
    input_pattern = os.getenv("INPUT_FILE_PATTERN")
    input_file = get_input_path(year, month)

    try:
        if s3_endpoint_url and input_pattern:
            options = {"client_kwargs": {"endpoint_url": s3_endpoint_url}}
            df = pd.read_parquet(input_file, storage_options=options)
            print(
                f"Data loaded from S3 input file {input_file} and storage options {options}"
            )
            return df

    except FileNotFoundError:
        print(f"File {input_file} not found on S3")

    print(f"Loading data from local file {filename}")
    df = pd.read_parquet(filename)
    print(f"Data loaded from local file {filename}")
    return df


def save_data(output_file, df):
    s3_endpoint_url = os.getenv("S3_ENDPOINT_URL")
    output_pattern = os.getenv("OUTPUT_FILE_PATTERN")

    print("s3_endpoint_url", s3_endpoint_url)
    print("output_pattern", output_pattern)

    if s3_endpoint_url and output_pattern:
        storage_options = {
            "client_kwargs": {"endpoint_url": s3_endpoint_url},
        }
        df.to_parquet(
            output_file,
            engine="pyarrow",
            index=False,
            storage_options=storage_options,
        )
        print(
            f"File saved to S3 as {output_file} with storage options {storage_options}"
        )
    else:
        df.to_parquet(output_file, engine="pyarrow", index=False)
        print(f"File saved locally as {output_file}")


def main(
    year: int,
    month: int,
) -> None:
    model_path = "homework/model.bin"
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    print(f"Loading model from {model_path}")
    with open("homework/model.bin", "rb") as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ["PULocationID", "DOLocationID"]
    df = read_data(
        year=year,
        month=month,
        filename=input_file,
    )

    df = prepare_data(
        df,
        year=year,
        month=month,
        categorical=categorical,
    )

    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    print("predicted mean duration:", y_pred.mean())

    df_result = pd.DataFrame()
    df_result["ride_id"] = df["ride_id"]
    df_result["predicted_duration"] = y_pred

    save_data(output_file, df_result)


if __name__ == "__main__":
    main(
        year=int(sys.argv[1]),
        month=int(sys.argv[2]),
    )
