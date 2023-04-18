import datetime
import logging
import os
import sys
import re

import boto3
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from pandas.errors import ParserError
import pygsheets


logging.basicConfig(level=logging.INFO)

DOCUMENT_ID = os.environ.get("DOCUMENT_ID")

S3 = boto3.resource("s3")
LOCALSTACK_URL = os.environ.get("LOCALSTACK_URL")

if LOCALSTACK_URL:
    S3 = boto3.resource("s3", endpoint_url=LOCALSTACK_URL)

S3_BUCKET = os.environ.get("S3_BUCKET")
S3_FOLDER = os.environ.get("S3_FOLDER")
S3_FOLDER_PRIVATE = os.environ.get("S3_FOLDER_PRIVATE")

DB_CONNECTION = os.environ.get("DB_CONNECTION")
DATABASE_NAME = os.environ.get("DATABASE_NAME")
GH_COLLECTION = os.environ.get("GH_COLLECTION")

REGEX_DATE = r"^202\d-[0,1]\d-[0-3]\d"

PRIVATE_FIELDS = ["Likely_infector"]

INCLUDE_FIELDS = [
    "ID",
    "Pathogen",
    "Country",
    "Country_ISO3",
    "Case_status",
    "Confirmation_method",
    "Healthcare_worker",
    "Occupation",
    "Gender",
    "Age",
    "Date_onset",
    "Date_onset_estimated",
    "Outcome",
    "Date_of_first_consult",
    "Date_death",
    "Date_recovered",
    "Location_Province",
    "Location_District",
    "Contact_with_case",
    "Source",
    "Source_II",
    "Source_III",
    "Date_entry",
    "Data_up_to",
]

# These fields are not in the public line list but are included in a private version
INCLUDE_FIELDS_PRIVATE = INCLUDE_FIELDS + PRIVATE_FIELDS
TODAY = datetime.datetime.today()


def setup_logger():
    h = logging.StreamHandler(sys.stdout)
    rootLogger = logging.getLogger()
    rootLogger.addHandler(h)
    rootLogger.setLevel(logging.DEBUG)


def get_mean_delay(
    df: pd.DataFrame, target_col: str, onset_col: str = "Date_onset"
) -> pd.Series:
    both = df[
        ~pd.isna(df[target_col])
        & ~pd.isna(df[onset_col])
        & df[target_col].astype(str).str.fullmatch(REGEX_DATE)
        & df[onset_col].astype(str).str.fullmatch(REGEX_DATE)
    ]
    try:
        target = pd.to_datetime(both[target_col])
        onset = pd.to_datetime(both[onset_col])
    except ParserError:
        logging.error("Error occured when parsing date from column")
        raise
    return datetime.timedelta(days=(target - onset).mean().days)


def get_data_with_estimated_onset(df: pd.DataFrame) -> pd.DataFrame:
    """Infers the onset date by using the mean delay between onset and
    hospitalisation or onset and death from records that have both. Then apply
    the delay to those where onset is missing but either hospitalisation or
    death date is known."""

    logging.info(
        "Mean delay to consult/hospitalization: %s",
        delay_to_consult_hosp := get_mean_delay(df, "Date_of_first_consult"),
    )
    logging.info(
        "Mean delay to death: %s", delay_death := get_mean_delay(df, "Date_death")
    )

    def estimate_onset(row):
        if isinstance(row.Date_onset, str) and re.match(REGEX_DATE, row.Date_onset):
            return pd.to_datetime(row.Date_onset)
        if isinstance(row.Date_death, str) and re.match(REGEX_DATE, row.Date_death):
            return pd.to_datetime(row.Date_death) - delay_death
        if isinstance(row.Date_of_first_consult, str) and re.match(
            REGEX_DATE, row.Date_of_first_consult
        ):
            return pd.to_datetime(row.Date_of_first_consult) - delay_to_consult_hosp
        return None

    df["Date_onset_estimated"] = df.apply(estimate_onset, axis=1)
    return df


def get_data() -> pd.DataFrame:
    logging.info("Getting data from Google Sheets")
    client = pygsheets.authorize(service_account_env_var="GOOGLE_CREDENTIALS")
    spreadsheet = client.open_by_key(DOCUMENT_ID)

    return spreadsheet[0].get_as_df(numerize=False)


def drop_private_fields(data):
    logging.info("Removing fields not in public list")
    return data[[f for f in INCLUDE_FIELDS if f in data.columns]]


def get_private_list(data: pd.DataFrame):
    logging.info("Cleaning data and running through pipeline")
    data_with_estimated_onset = get_data_with_estimated_onset(data[~pd.isna(data.ID)])
    return data_with_estimated_onset[
        [f for f in INCLUDE_FIELDS_PRIVATE if f in data_with_estimated_onset.columns]
    ]


def store_public_data(data: pd.DataFrame):
    assert set(data.columns) <= set(
        INCLUDE_FIELDS
    ), "Data has fields that should not be in public list"
    logging.info("Uploading data to S3 (public list)")
    csv_data = data.to_csv(index=False)
    try:
        S3.Object(S3_BUCKET, f"{S3_FOLDER}/{TODAY}.csv").put(Body=csv_data)
        S3.Object(S3_BUCKET, "latest.csv").put(Body=csv_data)
    except Exception:
        logging.exception("An exception occurred while trying to upload files")
        raise


def store_private_data(data: pd.DataFrame):
    assert set(data.columns) <= set(
        INCLUDE_FIELDS_PRIVATE
    ), "Data has fields that should not be in private list"
    logging.info("Uploading data to S3 (private list)")

    try:
        S3.Object(S3_BUCKET, f"{S3_FOLDER_PRIVATE}/latest.csv").put(
            Body=data.to_csv(index=False)
        )
    except Exception:
        logging.exception("An exception occurred while trying to upload files")
        raise


def data_to_db(data: pd.DataFrame):
    logging.info("Adding data to the database")
    try:
        client = MongoClient(DB_CONNECTION)
        database = client[DATABASE_NAME]
        for entry in data.to_dict("records"):
            find = {"ID": entry["ID"]}
            update = {"$set": entry}
            database[GH_COLLECTION].update_one(find, update, upsert=True)
    except PyMongoError:
        logging.exception("An error occurred while trying to insert data")
        raise


if __name__ == "__main__":
    setup_logger()
    logging.info("Starting Marburg 2023 data ingestion")
    private_list = get_private_list(get_data())
    public_list = drop_private_fields(private_list)
    store_public_data(public_list)
    store_private_data(private_list)
    data_to_db(private_list)
    logging.info("Work complete")
