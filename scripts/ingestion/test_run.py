import os
import math
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
from pymongo import MongoClient
import pytest

from run import (
    DB_CONNECTION,
    DATABASE_NAME,
    GH_COLLECTION,
    S3_BUCKET,
    LOCALSTACK_URL,
    get_private_list,
    drop_private_fields,
    store_private_data,
    store_public_data,
    data_to_db,
)

DATA = pd.read_csv(Path(__file__).with_name("test_data.csv"), dtype=str)

PRIVATE_LIST = get_private_list(DATA)
PUBLIC_LIST = drop_private_fields(PRIVATE_LIST)
PRIVATE_LIST_CSV = """ID,Case_status,Gender,Age,Date_onset,Date_onset_estimated,Date_of_first_consult,Date_death,Location_District,Data_up_to,Likely_infector
1,confirmed,male,50-55,2023-03-05,2023-03-05,,2023-03-09,Bata,2023-04-04,
2,probable,female,40-46,2023-02-06,2023-02-06,,2023-02-14,Bata,2023-04-04,1
3,confirmed,male,20,2023-02-19,2023-02-19,2023-02-25,,Nsoc Nsomo,2023-04-04,2
4,confirmed,female,99,2023-01-05,2023-01-05,,2023-01-11,Nsoc Nsomo,2023-04-04,
5,probable,male,65,NK,2023-01-13,,2023-01-19,Ebiebyin,2023-04-04,
6,confirmed,female,59,,2023-03-29,2023-04-02,,Ebiebyin,2023-04-04,
7,confirmed,male,0,2023-02-11,2023-02-11,2023-02-13,,Nsork,2023-04-04,
"""

PUBLIC_LIST_CSV = """ID,Case_status,Gender,Age,Date_onset,Date_onset_estimated,Date_of_first_consult,Date_death,Location_District,Data_up_to
1,confirmed,male,50-55,2023-03-05,2023-03-05,,2023-03-09,Bata,2023-04-04
2,probable,female,40-46,2023-02-06,2023-02-06,,2023-02-14,Bata,2023-04-04
3,confirmed,male,20,2023-02-19,2023-02-19,2023-02-25,,Nsoc Nsomo,2023-04-04
4,confirmed,female,99,2023-01-05,2023-01-05,,2023-01-11,Nsoc Nsomo,2023-04-04
5,probable,male,65,NK,2023-01-13,,2023-01-19,Ebiebyin,2023-04-04
6,confirmed,female,59,,2023-03-29,2023-04-02,,Ebiebyin,2023-04-04
7,confirmed,male,0,2023-02-11,2023-02-11,2023-02-13,,Nsork,2023-04-04
"""


def get_contents(file_name: str) -> str:
    s3 = boto3.resource("s3", endpoint_url=LOCALSTACK_URL)
    obj = s3.Object(S3_BUCKET, file_name)
    return obj.get()["Body"].read().decode("utf-8")


def get_db_records(collection: str) -> list[dict]:
    db = MongoClient(DB_CONNECTION)[DATABASE_NAME][collection]
    cursor = db.find({})
    return [record for record in cursor]


def test_get_private_list():
    assert PRIVATE_LIST.to_csv(index=False) == PRIVATE_LIST_CSV


def dict_equals(x: dict[str, Any], y: dict[str, Any]) -> bool:
    nan_x = [k for k in x if isinstance(x[k], float) and math.isnan(x[k])]
    nan_y = [k for k in x if isinstance(x[k], float) and math.isnan(x[k])]
    assert nan_x == nan_y, "Keys with null values do not match"
    return {k: v for k, v in x.items() if k not in nan_x} == {
        k: v for k, v in y.items() if k not in nan_y
    }


def test_get_public_list():
    assert PUBLIC_LIST.to_csv(index=False) == PUBLIC_LIST_CSV


@pytest.mark.skipif(
    not os.environ.get("DOCKERIZED", False),
    reason="Running e2e tests outside of mock environment disabled",
)
def test_store_data():
    store_public_data(PUBLIC_LIST)
    assert get_contents("latest.csv") == PUBLIC_LIST_CSV
    store_private_data(PRIVATE_LIST)
    assert get_contents("private/latest.csv") == PRIVATE_LIST_CSV


@pytest.mark.skipif(
    not os.environ.get("DOCKERIZED", False),
    reason="Running e2e tests outside of mock environment disabled",
)
def test_data_to_db():
    data_to_db(PRIVATE_LIST)
    db_records = get_db_records(GH_COLLECTION)
    for db_row, csv_row in zip(db_records, PRIVATE_LIST.to_dict("records")):
        del db_row["_id"]
        assert dict_equals(db_row, csv_row)
