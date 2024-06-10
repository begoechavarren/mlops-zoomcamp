from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

if "transformer" not in globals():
    from mage_ai.data_preparation.decorators import transformer
if "test" not in globals():
    pass


@transformer
def transform(df, *args, **kwargs):
    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df["duration"].dt.total_seconds() / 60
    filtered_df = df[(df["duration"] >= 1) & (df["duration"] <= 60)]
    locations_202101_dict = (
        filtered_df[["PULocationID", "DOLocationID"]]
        .astype(str)
        .to_dict(orient="records")
    )
    dv = DictVectorizer()
    X_train = dv.fit_transform(locations_202101_dict)
    y_train = filtered_df["duration"].values
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    print(f"Intercept: {lr.intercept_}")

    return dv, lr
