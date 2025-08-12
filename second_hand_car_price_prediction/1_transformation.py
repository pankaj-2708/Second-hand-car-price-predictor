import numpy as np
import pandas as pd
import pathlib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer
import yaml


def load_data(file_path):
    return pd.read_csv(file_path)


def select_random(df):
    return df.sample(1000, random_state=42)


def apply_transformation(df, desc, std):
    y = df[["vehical_price_in_lakh_inr"]]
    df.drop(columns="vehical_price_in_lakh_inr", axis=1, inplace=True)
    cat_columns = ["Insurance ", "Fuel Type ", "Transmission "]
    other_cols = []
    to_norm = [
        "Seats ",
        "Mileage in kmpl or km/kg",
        "Power in bhp",
        "new_vehical_price_in_lakh_inr",
        "Engine Displacement in cc",
        "Registration age",
        "Year of Manufacture age",
        "Kms Driven in 1000 km",
    ]
    for col in df.columns:
        if col not in to_norm:
            other_cols.append(col)
    df_arranged = pd.DataFrame()
    df_arranged[to_norm] = df[to_norm]
    df_arranged[cat_columns] = df[cat_columns]
    df_arranged[other_cols] = df[other_cols]

    clm = None

    if desc:
        clm = ColumnTransformer(
            [
                ("ins", OneHotEncoder(), [8, 9, 10]),
                ("log", FunctionTransformer(func=np.log1p), [2, 3, 4]),
                ("pow", PowerTransformer("box-cox"), [7]),
            ],
            remainder="passthrough",
        )
    else:
        clm = ColumnTransformer(
            [
                ("ins", OneHotEncoder(), [8, 9, 10]),
                ("log", FunctionTransformer(func=np.log1p), [2, 3, 4]),
                ("pow", KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile"), [7]),
            ],
            remainder="passthrough",
        )
    clm_std = None
    if std:
        clm_std = ColumnTransformer(
            [("norm", StandardScaler(), [i for i in range(8)])], remainder="passthrough"
        )
    else:
        clm_std = ColumnTransformer(
            [("norm", StandardScaler(), [i for i in range(7)])], remainder="passthrough"
        )

    ppl = Pipeline([("clm", clm), ("std", clm_std)])

    X = ppl.fit_transform(df_arranged)
    new_df = pd.DataFrame(X)
    new_df["target"] = y

    return new_df


def save_data(new_df, output_path):
    new_df.to_csv(output_path / "finalised.csv", index=False)


def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent
    data_path = home_dir / "data" / "processed" / "cleaned.csv"
    output_path = home_dir / "data" / "processed"
    params = None
    with open(home_dir / "params.yaml", "r") as f:
        params = yaml.safe_load(f)["transformation"]

    global df
    df = load_data(data_path)

    output_path.mkdir(parents=True, exist_ok=True)
    df = apply_transformation(output_path, params["discretize"], params["standardise"])
    save_data(df, output_path)


if __name__ == "__main__":
    main()
