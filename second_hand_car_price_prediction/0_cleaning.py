import pathlib
import numpy as np
import pandas as pd
import yaml
from datetime import datetime
import ast
import pickle
import warnings

warnings.filterwarnings("ignore")
df = None


def load_data(file_path):
    return pd.read_csv(file_path)


def handleNull(row):
    global df
    row["Power in bhp"] = df[
        (df["vehical_name"] == row["vehical_name"]) & (df["Power in bhp"].isnull() == False)
    ]["Power in bhp"].mean()
    return row


def handleNull2(row):
    global df
    row["Power in bhp"] = df[
        (df["new_vehical_price_in_lakh_inr"] > row["new_vehical_price_in_lakh_inr"] - 1)
        & (df["new_vehical_price_in_lakh_inr"] < row["new_vehical_price_in_lakh_inr"] + 1)
    ]["Power in bhp"].mean()
    return row


def handleNull3(row):
    row["Power in bhp"] = df[
        (df["new_vehical_price_in_lakh_inr"] > row["new_vehical_price_in_lakh_inr"] - 4)
        & (df["new_vehical_price_in_lakh_inr"] < row["new_vehical_price_in_lakh_inr"] + 4)
    ]["Power in bhp"].mean()
    return row


def handle_mil(mil):
    try:
        return float(mil.strip().split()[0])
    except:
        return mil


def handleNull_mileage(row):
    global i
    row["Mileage in kmpl or km/kg"] = df[
        (df["Fuel Type "] == row["Fuel Type "])
        & (df["Mileage in kmpl or km/kg"].isnull() == False)
        & (df["Registration Year "] > row["Registration Year "] - 3 * i)
        & (df["Registration Year "] < row["Registration Year "] + 3 * i)
        & (df["Year of Manufacture "] < row["Year of Manufacture "] + 2 * i)
        & (df["Year of Manufacture "] > row["Year of Manufacture "] - 2 * i)
        & (df["new_vehical_price_in_lakh_inr"] > row["new_vehical_price_in_lakh_inr"] - 1.5 * i)
        & (df["new_vehical_price_in_lakh_inr"] < row["new_vehical_price_in_lakh_inr"] + 1.5 * i)
        & (df["vehical_price_in_lakh_inr"] < row["vehical_price_in_lakh_inr"] + 0.25 * i)
        & (df["vehical_price_in_lakh_inr"] > row["vehical_price_in_lakh_inr"] - 0.25 * i)
    ]["Mileage in kmpl or km/kg"].mean()
    return row


i = 1


def fill_mileage(no_of_iter):
    global i
    while i < no_of_iter + 1:
        df.loc[df["Mileage in kmpl or km/kg"].isnull(), ["Mileage in kmpl or km/kg"]] = df[
            df["Mileage in kmpl or km/kg"].isnull()
        ].apply(handleNull_mileage, axis=1)["Mileage in kmpl or km/kg"]
        i += 1


def convert_price(prc):
    x = prc.replace("New Car Price", "").replace("₹", "").strip()
    crore = x.split()[-1] == "Crore"
    if crore:
        x = float(x.replace("Crore", "").strip()) * 100
    else:
        x = float(x.replace("Lakh", "").strip())
    return round(x, 2)


def convert_price2(prc):
    x = prc.replace("Make Your Offer", "").replace("₹", "").strip()
    crore = x.split()[1] == "Crore"
    thousand = x.split()[1] == "Thousand"
    if crore:
        x = float(x.replace("Crore", "").strip()) * 100
    elif thousand:
        x = float(x.replace("Thousand", "").strip()) / 100
    else:
        x = float(x.replace("Lakh", "").strip())
    return round(x, 2)


d = {
    "First": 1,
    "Second": 2,
    "Third": 3,
    "Fourth": 4,
    "Fifth": 5,
    "Sixth": 6,
    "Seventh": 7,
    "Eighth": 8,
    "Ninth": 9,
    "Tenth": 10,
}


def transform_owner(txt):
    return d[txt.replace("Owner", "").strip()]


def clean_data(output_path):

    # registration year and vehical_name
    df.loc[df["Registration Year "].isnull(), "Registration Year "] = (
        df[df["Registration Year "].isnull()]["vehical_name"]
        .apply(lambda x: x.split(" ")[0])
        .values
    )
    df["Registration Year "] = df["Registration Year "].apply(lambda x: int(x.strip().split()[-1]))
    df["company_name"] = df["vehical_name"].apply(lambda x: x.split(" ")[1])
    df["model_detail"] = df["vehical_name"].apply(lambda x: " ".join(x.split(" ")[2:]))
   
    df.dropna(subset="Year of Manufacture ", inplace=True)
    df["Year of Manufacture "] = df["Year of Manufacture "].apply(lambda x: int(x))
    

    # insurance
    df["Insurance "] = np.where(df["Insurance "] == "-", "No Insurance", df["Insurance "])

    # seats
    df.dropna(subset="Seats ", inplace=True)

    # kms_driven
    df["Kms Driven "] = df["Kms Driven "].apply(
        lambda x: int(x.replace(",", "").strip().split(" ")[0])
    )
    

    # engine displacement
    df["Engine Displacement "] = df["Engine Displacement "].fillna("-1")
    df["Engine Displacement "] = df["Engine Displacement "].apply(
        lambda x: int(x.strip().split(" ")[0])
    )
    df["Engine Displacement in cc"] = np.where(
        df["Engine Displacement "] == -1,
        df["Engine Displacement "].mean(),
        df["Engine Displacement "],
    )

    # owner ship
    df["Ownership "] = df["Ownership "].fillna("First Owner")

    df.drop(columns=["Fuel ", "Engine "], axis=1, inplace=True)

    # new_vehical_price
    df["new_vehical_price"] = df["new_vehical_price"].apply(convert_price)
    df.rename(columns={"new_vehical_price": "new_vehical_price_in_lakh_inr"}, inplace=True)

    # vehical price
    df["vehical_price"].apply(
        lambda x: x.replace("Make Your Offer", "").replace("₹", "").strip().split()[1]
    ).value_counts()
    df["vehical_price"] = df["vehical_price"].apply(convert_price2)
    df.rename(columns={"vehical_price": "vehical_price_in_lakh_inr"}, inplace=True)

    # power
    df.loc[df["Power "].isnull() == False, "Power "] = df[df["Power "].isnull() == False][
        "Power "
    ].apply(lambda x: float(x.strip().split(" ")[0]))
    df.rename(columns={"Power ": "Power in bhp"}, inplace=True)
    df.loc[df["Power in bhp"].isnull(), "Power in bhp"] = df[df["Power in bhp"].isnull()].apply(
        handleNull, axis=1
    )
    df.loc[df["Power in bhp"].isnull(), "Power in bhp"] = df[df["Power in bhp"].isnull()].apply(
        handleNull2, axis=1
    )
    df.loc[df["Power in bhp"].isnull(), "Power in bhp"] = df[df["Power in bhp"].isnull()].apply(
        handleNull3, axis=1
    )
    df.dropna(subset=["Power in bhp"], inplace=True)

    # drive type
    df.drop(columns=["Drive Type "], axis=1, inplace=True)

    # mileage
    df["Mileage "] = df["Mileage "].apply(handle_mil)
    df.rename(columns={"Mileage ": "Mileage in kmpl or km/kg"}, inplace=True)
    fill_mileage(10)
    df.dropna(subset=["Mileage in kmpl or km/kg"], inplace=True)

    # RTO
    df["RTO "] = df["RTO "].fillna("Others")
    

    # seats
    df["Seats "] = df["Seats "].apply(lambda x: int(x.replace("Seats", "").strip()))

    # no of owners
    df["No of owners"] = df["Ownership "].apply(transform_owner)


    df.drop(
        columns=[
            "Ownership ",
            'vehical_name',
            "Transmission .1",
            "Engine Displacement "
        ],
        axis=1,
        inplace=True,
    )


def save_data(output_path):
    df.to_csv(output_path / "cleaned.csv", index=False)


def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent

    data_path = home_dir / "data" / "raw" / "car_details.csv"
    output_path = home_dir / "data" / "processed"

    global df
    df = load_data(data_path)

    output_path.mkdir(parents=True, exist_ok=True)
    clean_data(output_path)
    save_data(output_path)


if __name__ == "__main__":
    main()
