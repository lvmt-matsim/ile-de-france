import geopandas as gpd
import pandas as pd
import shapely.geometry as geo
import os, datetime, json

def configure(context):
    context.stage("synthesis.population.enriched")
    context.config("output_path")
    context.config("output_prefix", "ile_de_france_")

def validate(context):
    output_path = context.config("output_path")

    if not os.path.isdir(output_path):
        raise RuntimeError("Output directory must exist: %s" % output_path)

def execute(context):
    output_path = context.config("output_path")
    output_prefix = context.config("output_prefix")

    # Prepare households
    df_households = context.stage("synthesis.population.enriched").rename(
        columns = { "household_income": "income" }
    ).drop_duplicates("household_id")

    df_households = df_households[[
        "household_id",
        "car_availability", "bike_availability",
        "number_of_vehicles", "number_of_bikes",
        "income",
        "census_household_id",
        "housing_type","household_type", "parking","parking_at_workplace","commuting_dist_egt", "achlr",
        "ENERGV1_egt", "APMCV1_egt","ENERGV2_egt", "APMCV2_egt","ENERGV3_egt", "APMCV3_egt","ENERGV4_egt", "APMCV4_egt"


    ]]

    df_households.to_csv("%s/%shouseholds_test.csv" % (output_path, output_prefix), sep = ";", index = None)

  # Prepare persons
    df_persons = context.stage("synthesis.population.enriched").rename(
        columns = { "has_license": "has_driving_license" }
    )

    df_persons = df_persons[[
        "person_id", "household_id",
        "age", "employed", "sex", "socioprofessional_class",
        "has_driving_license", "has_pt_subscription",
        "census_person_id", "hts_id"
    ]]

    df_persons.to_csv("%s/%spersons_test.csv" % (output_path, output_prefix), sep = ";", index = None)