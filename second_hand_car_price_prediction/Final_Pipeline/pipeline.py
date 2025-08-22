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
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
import cloudpickle
import dill
warnings.filterwarnings("ignore")


class preprocess(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        with open("../../data/transformed/rto.pkl",'rb') as f:
            self.all_rto = pickle.load(f)
        with open("../../data/transformed/companies.pkl",'rb') as f:
            self.all_companies = pickle.load(f)
        with open("../../data/transformed/car_models.pkl",'rb') as f:
            self.all_models = pickle.load(f)
        with open("../../data/transformed/add_features.pkl",'rb') as f:
            self.add_features = pickle.load(f)
        return self

    def transform(self, df,y=None):
        df["Registration age"] = pd.to_datetime(df["Registration Year "], format="%Y").apply(
            lambda x: round((datetime.now().date() - x.date()).days / 365, 2)
        )
        df["Year of Manufacture age"] = pd.to_datetime(
            df["Year of Manufacture "], format="%Y"
        ).apply(lambda x: round((datetime.now().date() - x.date()).days / 365, 2))
        df["Kms Driven in 1000 km"] = df["Kms Driven "] / 1000
        df=df.copy()
        # print("rto",len(self.all_rto))
        # print("all_companies",len(self.all_companies))
        # print("all_models",len(self.all_models))
        # print("add_features",len(self.add_features))
        # print(df.shape)
        for rto in self.all_rto:
            df[rto] = [0 for i in range(df.shape[0])]

        for i in range(df.shape[0]):
            df.loc[i, df.loc[i, "RTO "]] = 1


        # print(df.shape)
        for comp in self.all_companies:
            df[comp] = [0 for i in range(df.shape[0])]

        for i in range(df.shape[0]):
            df.loc[i, df.loc[i, "company_name"]] = 1


        # print(df.shape)
        for model in df.loc[:,"model_detail"]:
            if model not in self.all_models:
                df.loc[:,"model_detail"]="Others"
                
        for model in self.all_models:
            df[model] = [0 for i in range(df.shape[0])]

        for i in range(df.shape[0]):
            df.loc[i, df.loc[i, "model_detail"]] = 1


        # print(df.shape)
        # unexpected = set()
        # for i in range(df.shape[0]):
        #     for feature in df.loc[i, "other_features"]:
        #         if feature not in self.add_features:
        #             unexpected.add(feature)

        # print("Unexpected other_features:", unexpected)
        
        df["other_features"] = df["other_features"].apply(ast.literal_eval)
        for feature in self.add_features:
            if feature in self.add_features:
                df[feature] = [0 for i in range(df.shape[0])]

        for i in range(df.shape[0]):
            for feature in df.loc[i, "other_features"]:
                if feature in self.add_features:
                    df.loc[i, df.loc[i, feature]] = 1
        if 0 in df.columns:
            df=df.drop(columns=[0])
        # for col in df.columns:
            # print(col,type(col))
        df.drop(
            columns=[
                "Registration Year ",
                "Year of Manufacture ",
                "Kms Driven ",
                "RTO ",
                "company_name",
                "model_detail",
                "other_features"
            ],
            inplace=True,
            axis=1,
        )
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

        df_arranged.columns = df_arranged.columns.astype(str)

        # print(df.shape)
        # print(df_arranged.columns[:11])
        # df.to_csv("temp.csv",index=False)
        return df_arranged



clm = ColumnTransformer(
    [
        ("ins", OneHotEncoder(), [8, 9, 10]),
        ("log", FunctionTransformer(func=np.log1p), [2, 3, 4]),
         ("pow", KBinsDiscretizer(n_bins=15, encode="ordinal", strategy="quantile"), [7])
    ],
    remainder="passthrough",
)

clm_std = ColumnTransformer(
    [("norm", StandardScaler(), [i for i in range(7)])], remainder="passthrough"
)


ppl = Pipeline(
    [
        ("preprocess", preprocess()),
        ("clm", clm),
        ("clm_std", clm_std),
        ("model",RandomForestRegressor())
    ]
)

df=pd.read_csv('./data/processed/cleaned.csv')

df.columns = df.columns.astype(str)
X=df.drop(columns=["vehical_price_in_lakh_inr"],axis=1)
y=df['vehical_price_in_lakh_inr']
ppl.fit(X,y)
with open('./data/transformed/ppl.pkl',"wb") as f:
    cloudpickle.dump(ppl,f)
