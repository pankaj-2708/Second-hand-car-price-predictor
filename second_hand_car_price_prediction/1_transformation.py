import numpy as np
import pandas as pd
import pathlib
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer
import yaml
import ast
import pickle
import warnings

warnings.filterwarnings("ignore")

def load_data(file_path):
    return pd.read_csv(file_path)


def select_random(df):
    return df.sample(1000, random_state=42)


def apply_transformation(df, desc, std,output_path, min_rto_count, min_model_count):
    df["Registration age"] = pd.to_datetime(df["Registration Year "], format="%Y").apply(
        lambda x: round((datetime.now().date() - x.date()).days / 365, 2)
    )
    
    df["Year of Manufacture age"] = pd.to_datetime(df["Year of Manufacture "], format="%Y").apply(
        lambda x: round((datetime.now().date() - x.date()).days / 365, 2)
    )
    df["Kms Driven in 1000 km"] = df["Kms Driven "] / 1000
    val_cnt = df["RTO "].value_counts()
    for val, ind in zip(val_cnt.values, val_cnt.index):
        if val < min_rto_count:
            val_cnt.loc["Others"] += val
            val_cnt.drop(ind, inplace=True)

    # for streamlit webapp
    df.reset_index(drop=True, inplace=True)
    with open(output_path / "rto.pkl", "wb") as f:
        pickle.dump(val_cnt.index, f)
    for rto in val_cnt.index:
        df[rto] = [0 for i in range(df.shape[0])]
    for rto in val_cnt.index:
        for i in range(df.shape[0]):
            if rto == df.loc[i, "RTO "]:
                df.loc[i, rto] = 1

    companies = []
    for i in df["company_name"].values:
        companies.append(i)
    companies = list(set(companies))

    # for streamlit webapp
    with open(output_path / "companies.pkl", "wb") as f:
        pickle.dump(companies, f)
    for comp in companies:
        df[comp] = [0 for i in range(df.shape[0])]

    for comp in companies:
        for i in range(df.shape[0]):
            if comp == df.loc[i, "company_name"]:
                df.loc[i, comp] = 1
                
    

    # models
    models = {}
    for i in df["model_detail"].values:
        models[i] = models.get(i, 0) + 1

    major_models = ["other_model"]
    for model in models.keys():
        if models[model] > min_model_count:
            major_models.append(model)

    # for streamlit webapp
    with open(output_path / "car_models.pkl", "wb") as f:
        pickle.dump(major_models, f)
    for model in major_models:
        df[model] = [0 for i in range(df.shape[0])]
    df["other_model"] = [0 for i in range(df.shape[0])]

    for model in major_models:
        for i in range(df.shape[0]):
            if model == df.loc[i, "model_detail"]:
                df.loc[i, model] = 1
            else:
                df.loc[i, "other_model"] = 1

    # other features
    df["other_features"] = df["other_features"].apply(ast.literal_eval)
    add_features = []
    for i in df["other_features"].values:
        add_features += i
    add_features = list(set(add_features))

    # for streamlit webapp
    with open(output_path / "add_features.pkl", "wb") as f:
        pickle.dump(add_features, f)
    for feature in add_features:
        df[feature] = [0 for i in range(df.shape[0])]
    for feature in add_features:
        for i in range(df.shape[0]):
            if feature in df.loc[i, "other_features"]:
                df.loc[i, feature] = 1
    with open(output_path / "fuel_types.pkl", "wb") as f:
        pickle.dump(add_features, f)
        
    with open(output_path / "insurance_values.pkl", "wb") as f:
        pickle.dump(df['Insurance '].values, f)
        
    with open(output_path / "fuel_types.pkl", "wb") as f:
        pickle.dump(df['Fuel Type '].values, f)

    with open(output_path / "transmission_values.pkl", "wb") as f:
        pickle.dump(df['Transmission '].values, f)
    
    df.drop(
        columns=[
            "Registration Year ",
            "Year of Manufacture ",
            "Kms Driven ",
            "RTO ",
            "other_features",
            "company_name",
            "model_detail"
        ],
        axis=1,
        inplace=True,
    )
    # drop kms driven,reg year,year of manufacture,RTO,companies
    # path_=str(output_path / "pipeline_training.csv")
    # df.to_csv(path_,index=False)
    y = df[["vehical_price_in_lakh_inr"]]
    df.drop(columns=["vehical_price_in_lakh_inr"], axis=1, inplace=True)
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
        if col not in to_norm and col not in cat_columns:
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
                ("pow", KBinsDiscretizer(n_bins=15, encode="ordinal", strategy="quantile"), [7]),
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
    output_path = home_dir / "data" / "transformed"
    params = None
    with open(home_dir / "params.yaml", "r") as f:
        params = yaml.safe_load(f)["transformation"]

    global df
    df = load_data(data_path)

    output_path.mkdir(parents=True, exist_ok=True)
    df = apply_transformation(df, params["discretize"], params["standardise"], output_path,params["min_rto_count"], params["min_model_count"])
    save_data(df, output_path)


if __name__ == "__main__":
    main()
