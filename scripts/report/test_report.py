import io
import os
import datetime
from pathlib import Path

import json
import pytest
import pandas as pd

from report import (
    S3_BUCKET,
    S3_BUCKET_REPORT,
    store_s3,
    S3,
    get_age_bin_data,
    get_age_bins,
    get_epicurve,
    get_delays,
    get_counts,
    name_bin,
    build,
)

CSV_DATA = Path(__file__).with_name("test_data.csv").read_text(encoding="utf-8")
DATA = pd.read_csv(io.StringIO(CSV_DATA))


@pytest.mark.parametrize(
    "age,expected_range",
    [("50-55", range(6, 7)), ("0", range(0, 1)), ("65", range(7, 8))],
)
def test_get_age_bins(age, expected_range):
    "Returns age bin sequence range from age string in format start-end"
    assert get_age_bins(age) == expected_range


@pytest.mark.parametrize("bin_idx,expected", [(0, "0"), (1, "1-9"), (9, "80+")])
def test_name_bin(bin_idx, expected):
    assert name_bin(bin_idx) == expected


def test_get_age_bin_data():
    expected = pd.DataFrame(
        [
            dict(Bin="0", Gender="male", N=1.0),
            dict(Bin="20-29", Gender="male", N=1.0),
            dict(Bin="50-59", Gender="female", N=1.0),
            dict(Bin="50-59", Gender="male", N=1.0),
            dict(Bin="80+", Gender="female", N=1.0),
        ]
    )
    assert get_age_bin_data(DATA).equals(expected)


@pytest.mark.parametrize(
    "column,expected_delay_series",
    [("Date_death", [4, 8, 6]), ("Date_of_first_consult", [6, 2])],
)
def test_get_delays(column, expected_delay_series):
    assert list(get_delays(DATA, column).dt.days) == expected_delay_series


def test_get_epicurve():
    dates = [
        "2023-" + md
        for md in ["01-05", "01-13", "02-06", "02-11", "02-19", "03-05", "03-29"]
    ]
    expected = pd.DataFrame(
        {"Date_onset": dates, "Cumulative_cases": list(range(1, 8))}
    )
    expected["Date_onset"] = pd.to_datetime(expected.Date_onset)
    assert get_epicurve(DATA).equals(expected)


def test_get_counts():
    assert get_counts(DATA) == {
        "n_confirmed": 5,
        "n_probable": 2,
        "date": "2023-04-04",
        "pc_valid_age_gender_in_confirmed": 100,
    }


def get_contents(file_name: str, bucket_name: str = S3_BUCKET) -> str:
    obj = S3.Object(bucket_name, file_name)
    return obj.get()["Body"].read().decode("utf-8")


@pytest.mark.skipif(
    not os.environ.get("DOCKERIZED", False),
    reason="Running e2e tests outside of mock environment disabled",
)
def test_e2e():
    store_s3(CSV_DATA, f"archive/{datetime.datetime.today()}.csv")
    store_s3(CSV_DATA, "latest.csv")

    assert get_contents("latest.csv") == CSV_DATA
    build(S3_BUCKET)
    assert get_contents("index.html", S3_BUCKET_REPORT)
    assert json.loads(get_contents("metadata.json", S3_BUCKET_REPORT)) == dict(
        n_confirmed=5,
        n_probable=2,
        pc_valid_age_gender_in_confirmed=100,
        date="2023-04-04",
    )