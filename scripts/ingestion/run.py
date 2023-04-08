from datetime import datetime
import io
import csv
import logging
import os
import sys

import boto3
from pymongo import MongoClient
from pymongo.errors import PyMongoError
import pygsheets


DOCUMENT_ID = os.environ.get("DOCUMENT_ID")

S3 = boto3.resource("s3")
LOCALSTACK_URL = os.environ.get("LOCALSTACK_URL")

if LOCALSTACK_URL:
    S3 = boto3.resource("s3", endpoint_url=LOCALSTACK_URL)

S3_BUCKET = os.environ.get("S3_BUCKET")
S3_FOLDER = os.environ.get("S3_FOLDER")

DB_CONNECTION = os.environ.get("DB_CONNECTION")
DATABASE_NAME = os.environ.get("DATABASE_NAME")
GH_COLLECTION = os.environ.get("GH_COLLECTION")

INCLUDE_FIELDS = [
    "ID",
    "Pathogen",
    "Country",
    "Country_ISO3",
    "Case_status",
    "Confirmation_method",
    "Occupation",
    "Gender",
    "Age",
    "Date_onset",
    "Outcome",
    "Date_of_first_consult",
    "Date_death",
    "Date_recovered",
    "Location_District",
    "Location_Subdistrict",
    "Contact_with_case",
    "Contact_ID",
    "Source",
    "Date_entry",
    "Data_up_to",
]
TODAY = datetime.today()


def setup_logger():
    h = logging.StreamHandler(sys.stdout)
    rootLogger = logging.getLogger()
    rootLogger.addHandler(h)
    rootLogger.setLevel(logging.DEBUG)


def get_data():
    logging.info("Getting data from Google Sheets")
    client = pygsheets.authorize(service_account_env_var="GOOGLE_CREDENTIALS")
    spreadsheet = client.open_by_key(DOCUMENT_ID)

    return spreadsheet[0].get_all_records()


def clean_data_streamed(data):
    logging.info("Cleaning data")
    for c in data:
        if c["ID"] is None or (isinstance(c["ID"], str) and c["ID"].strip() == ""):
            continue
        yield {field: c.get(field) for field in set(INCLUDE_FIELDS) & set(c.keys())}


def format_data(data):
    logging.info("Formatting data")
    buf = io.StringIO()
    all_keys = set(sum((list(c.keys()) for c in data), []))
    writer = csv.DictWriter(
        buf, fieldnames=[k for k in INCLUDE_FIELDS if k in all_keys]
    )
    writer.writeheader()
    for row in data:
        writer.writerow(row)
    return buf.getvalue()


def store_data(csv_data):
    logging.info("Uploading data to S3")
    try:
        S3.Object(S3_BUCKET, f"{S3_FOLDER}/{TODAY}.csv").put(Body=csv_data)
        S3.Object(S3_BUCKET, "latest.csv").put(Body=csv_data)
    except Exception:
        logging.exception("An exception occurred while trying to upload files")
        raise


def data_to_db(data):
    logging.info("Adding data to the database")
    try:
        client = MongoClient(DB_CONNECTION)
        database = client[DATABASE_NAME]
        for entry in data:
            find = {"ID": entry["ID"]}
            update = {"$set": entry}
            database[GH_COLLECTION].update_one(find, update, upsert=True)
    except PyMongoError:
        logging.exception("An error occurred while trying to insert data")
        raise


if __name__ == "__main__":
    setup_logger()
    logging.info("Starting Marburg 2023 data ingestion")
    data = get_data()
    data = list(clean_data_streamed(data))
    csv_data = format_data(data)
    store_data(csv_data)
    data_to_db(data)
    logging.info("Work complete")
