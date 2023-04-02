import logging
import os
from time import sleep

import boto3
import requests


LOCALSTACK_URL = os.environ.get("AWS_ENDPOINT", "http://localstack:4566")
S3_BUCKET = os.environ.get("S3_BUCKET", "source")
S3_BUCKET_REPORT = os.environ.get("S3_BUCKET_REPORT", "report")

MAX_ATTEMPTS = 42
WAIT_TIME = 5


def wait_for_localstack():
    logging.info("Waiting for localstack")
    healthcheck_url = "".join([LOCALSTACK_URL, "/health"])
    counter = 0
    while counter < MAX_ATTEMPTS:
        try:
            response = requests.get(healthcheck_url)
            s3_status = response.json().get("services", {}).get("s3")
            if s3_status in ["available", "ready"]:
                return
        except requests.exceptions.ConnectionError:
            pass
        counter += 1
        sleep(WAIT_TIME)
    raise Exception("Localstack not available")


def create_bucket(bucket_name: str) -> None:
    logging.info(f"Creating S3 bucket {bucket_name}")
    s3_client = boto3.client("s3", endpoint_url=LOCALSTACK_URL)
    s3_client.create_bucket(Bucket=bucket_name)


if __name__ == "__main__":
    logging.info("Starting local/testing setup script")
    wait_for_localstack()
    create_bucket(S3_BUCKET)
    create_bucket(S3_BUCKET_REPORT)
    logging.info("Done")
