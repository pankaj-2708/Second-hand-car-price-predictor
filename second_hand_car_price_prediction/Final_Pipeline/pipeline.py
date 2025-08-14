import pickle
import ast
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore")


class preprocess(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        df["Registration age"] = pd.to_datetime(df["Registration Year"], format="%Y").apply(
            lambda x: round((datetime.now().date() - x.date()).days / 365, 2)
        )
        df["Year of Manufacture age"] = pd.to_datetime(
            df["Year of Manufacture"], format="%Y"
        ).apply(lambda x: round((datetime.now().date() - x.date()).days / 365, 2))
        df["Kms Driven in 1000 km"] = df["Kms Driven"] / 1000

        with open("../../data/processed/rto.pkl",'rb') as f:
            all_rto = pickle.load(f)

        for rto in all_rto:
            df[rto] = [0 for i in range(df.shape[0])]

        for i in range(df.shape[0]):
            df.loc[i, df.loc[i, "rto"]] = 1

        with open("../../data/processed/companies.pkl",'rb') as f:
            all_companies = pickle.load(f)

        for comp in all_companies:
            df[comp] = [0 for i in range(df.shape[0])]

        for i in range(df.shape[0]):
            df.loc[i, df.loc[i, "company"]] = 1

        with open("../../data/processed/car_models.pkl",'rb') as f:
            all_models = pickle.load(f)

        for model in all_models:
            df[model] = [0 for i in range(df.shape[0])]

        for i in range(df.shape[0]):
            df.loc[i, df.loc[i, "model"]] = 1

        with open("../../data/processed/add_features.pkl",'rb') as f:
            add_features = pickle.load(f)

        df["add_features"] = df["add_features"].apply(ast.literal_eval)
        for feature in add_features:
            df[feature] = [0 for i in range(df.shape[0])]

        for i in range(df.shape[0]):
            for feature in df.loc[i, "add_features"]:
                df.loc[i, df.loc[i, feature]] = 1

        df.drop(
            columns=[
                "Registration Year",
                "Year of Manufacture",
                "Kms Driven",
                "rto",
                "company",
                "model",
                "add_features",
            ],
            inplace=True,
            axis=1,
        )

        cat_columns = ["Insurance", "Fuel Type", "Transmission"]
        other_cols = []
        to_norm = [
            "Seats",
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

        df_arranged.columns = df_arranged.columns.astype(str)
        print(df.shape)
        return df_arranged



clm = ColumnTransformer(
    [
        ("ins", OneHotEncoder(), [8, 9, 10]),
        ("log", FunctionTransformer(func=np.log1p), [2, 3, 4]),
        ("pow", PowerTransformer("box-cox"), [7]),
    ],
    remainder="passthrough",
)

clm_std = ColumnTransformer(
    [("norm", StandardScaler(), [i for i in range(8)])], remainder="passthrough"
)


ppl = Pipeline(
    [
        ("preprocess", preprocess()),
        ("clm", clm),
        ("clm_std", clm_std)
    ]
)

df=pd.read_csv('../../data/frontend/demo.csv')

df.columns = df.columns.astype(str)

ppl.fit(df,np.array([0]))
x=ppl.transform(df)
print(x.shape)
with open('../../data/frontend/ppl.pkl',"wb") as f:
    pickle.dump(ppl,f)