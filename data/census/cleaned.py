from tqdm import tqdm
import pandas as pd
import numpy as np
import data.hts.hts as hts

"""
This stage cleans the French population census:
  - Assign new unique integer IDs to households and persons
  - Clean up spatial information and sociodemographic attributes
"""

def configure(context):
    context.stage("data.census.raw")
    context.stage("data.spatial.codes")

def execute(context):
    df = pd.read_hdf("%s/census.hdf" % context.path("data.census.raw"))

    # Construct household IDs for persons with NUMMI != Z
    df_household_ids = df[["CANTVILLE", "NUMMI"]]
    df_household_ids = df_household_ids[df_household_ids["NUMMI"] != "Z"]
    df_household_ids["temporary"] = df_household_ids["CANTVILLE"] + df_household_ids["NUMMI"]
    df_household_ids = df_household_ids.drop_duplicates("temporary")
    df_household_ids["household_id"] = np.arange(len(df_household_ids))
    df = pd.merge(df, df_household_ids, on = ["CANTVILLE", "NUMMI"], how = "left")

    # Fill up undefined household ids (those where NUMMI == Z)
    f = np.isnan(df["household_id"])
    df.loc[f, "household_id"] = np.arange(np.count_nonzero(f)) + df["household_id"].max()
    df["household_id"] = df["household_id"].astype(np.int)

    # Put person IDs
    df["person_id"] = np.arange(len(df))

    # Sorting
    df = df.sort_values(by = ["household_id", "person_id"])

    # Spatial information
    df["departement_id"] = df["DEPT"].astype("category")

    df["commune_id"] = df["IRIS"].str[:5]
    f_undefined = df["commune_id"].str.contains("Z")
    df.loc[f_undefined, "commune_id"] = "undefined"
    df["commune_id"] = df["commune_id"].astype("category")

    df["iris_id"] = df["IRIS"]
    f_undefined = df["iris_id"].str.contains("Z") | df["iris_id"].str.contains("X")
    df.loc[f_undefined, "iris_id"] = "undefined"
    df["iris_id"] = df["iris_id"].astype("category")

    # Verify with requested codes
    df_codes = context.stage("data.spatial.codes")

    excess_communes = set(df["commune_id"].unique()) - set(df_codes["commune_id"].unique())
    if not excess_communes == {"undefined"}:
        raise RuntimeError("Found additional communes: %s" % excess_communes)

    excess_iris = set(df["iris_id"].unique()) - set(df_codes["iris_id"].unique())
    if not excess_iris == {"undefined"}:
        raise RuntimeError("Found additional IRIS: %s" % excess_iris)

    # Age
    df["age"] = df["AGED"].apply(lambda x: "0" if x == "000" else x.lstrip("0")).astype(np.int)

    # Clean COUPLE
    df["couple"] = df["COUPLE"] == "1"

    # Clean TRANS
    df.loc[df["TRANS"] == "1", "commute_mode"] = np.nan
    df.loc[df["TRANS"] == "2", "commute_mode"] = "walk"
    df.loc[df["TRANS"] == "3", "commute_mode"] = "bike"
    df.loc[df["TRANS"] == "4", "commute_mode"] = "car"
    df.loc[df["TRANS"] == "5", "commute_mode"] = "pt"
    df.loc[df["TRANS"] == "Z", "commute_mode"] = np.nan
    df["commute_mode"] = df["commute_mode"].astype("category")

    # Weight
    df["weight"] = df["IPONDI"].astype(np.float)

    # Clean SEXE
    df.loc[df["SEXE"] == "1", "sex"] = "male"
    df.loc[df["SEXE"] == "2", "sex"] = "female"
    df["sex"] = df["sex"].astype("category")

    # Clean employment
    df["employed"] = df["TACT"] == "11"

    # Studies
    df["studies"] = df["ETUD"] == "1"

    # Number of vehicles
    df["number_of_vehicles"] = df["VOIT"].apply(
        lambda x: str(x).replace("Z", "0").replace("X", "0")
    ).astype(np.int)

    df["number_of_vehicles"] += df["DEROU"].apply(
        lambda x: str(x).replace("U", "0").replace("Z", "0").replace("X", "0")
    ).astype(np.int)

    # Household size
    df_size = df[["household_id"]].groupby("household_id").size().reset_index(name = "household_size")
    df = pd.merge(df, df_size)

    # Socioprofessional category
    df["socioprofessional_class"] = df["CS1"].astype(np.int)

    # Place of work or education
    df["work_outside_region"] = df["ILT"].isin(("4", "5", "6"))
    df["education_outside_region"] = df["ILETUD"].isin(("4", "5", "6"))

    # Consumption units
    df = pd.merge(df, hts.calculate_consumption_units(df), on = "household_id")

    #  Clean TYPL : housing type,  BYIN
    df.loc[df["TYPL"] == "1", "housing_type"] = "house"
    df.loc[df["TYPL"] == "2", "housing_type"] = "flat"
    df.loc[df["TYPL"] == "3", "housing_type"] = "others"
    df.loc[df["TYPL"] == "4", "housing_type"] = "others"
    df.loc[df["TYPL"] == "5", "housing_type"] = "others"
    df.loc[df["TYPL"] == "6", "housing_type"] = "others"
    df.loc[df["TYPL"] == "Z", "housing_type"] = "others"
    df["housing_type"] = df["housing_type"].astype("category")

    #  Clean SFM : household type,  BYIN
    df.loc[df["SFM"] == "11", "household_type"] = "SM"
    df.loc[df["SFM"] == "12", "household_type"] = "SW"
    df.loc[df["SFM"] == "21", "household_type"] = "MFF"
    df.loc[df["SFM"] == "22", "household_type"] = "MFM"
    df.loc[df["SFM"].isin(["30"]), "household_type"] = "CWOC"
    df.loc[df["SFM"].isin(["31","32","33","34"]), "household_type"] = "CWC"
    df.loc[df["SFM"].isin(["40","51","52","53","54","61","62","70","ZZ"]), "household_type"] = "others"
    df["household_type"] = df["household_type"].astype("category")

    # Clean GARL: presence of home parking, BYIN
    df.loc[df["GARL"] == "1", "parking"] = 1
    df.loc[df["GARL"] == "2", "parking"] = 0
    df.loc[df["GARL"] == "Z", "parking"] = 0
    df["parking"] = df["parking"].astype(np.int)

    # ACHLR: Période regroupée d'achèvement de la construction de la maison ou de l'immeuble BYIN
    df["achlr"] = df["ACHLR"].astype("category")


    return df[[
        "person_id", "household_id", "weight",
        "iris_id", "commune_id", "departement_id",
        "age", "sex", "couple",
        "commute_mode", "employed",
        "studies", "number_of_vehicles", "household_size",
        "work_outside_region", "education_outside_region",
        "consumption_units", "socioprofessional_class",
        "housing_type", "household_type", "parking", "achlr"
    ]]
