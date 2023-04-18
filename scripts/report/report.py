"""
Briefing report generator for Marburg 2023 outbreak
"""
import os
import re
import sys
import json
import tempfile
import logging
import datetime
from pathlib import Path
from typing import Any
from functools import cache

import boto3
import chevron
import pandas as pd
import numpy as np
from dateutil.parser import ParserError
import plotly.graph_objects as go
import plotly.io
import plotly.express as px
from plotly.subplots import make_subplots


pd.options.mode.chained_assignment = None

REGEX_DATE = r"^202\d-[0,1]\d-[0-3]\d"
OVERRIDES = {}
FONT = "Inter"
TITLE_FONT = "mabry-regular-pro"
LEGEND_FONT_SIZE = 14
AGE_BINS = [
    (0, 0),
    (1, 9),
    (10, 19),
    (20, 29),
    (30, 39),
    (40, 49),
    (50, 59),
    (60, 69),
    (70, 79),
    (80, 120),
]

S3 = boto3.resource("s3", endpoint_url=os.getenv("LOCALSTACK_URL"))
S3_BUCKET = os.getenv("S3_BUCKET", "marburg-gh")
S3_BUCKET_REPORT = os.getenv("S3_BUCKET_REPORT", "www.marburg.global.health")
S3_AGGREGATES = os.getenv("S3_AGGREGATES", "marburg-aggregates")

GREEN_PRIMARY_COLOR = "#0E7569"
BLUE_PRIMARY_COLOR = "#007AEC"
PRIMARY_COLOR = GREEN_PRIMARY_COLOR
SECONDARY_COLOR = "#00C6AF"
BG_COLOR = "#ECF3F0"
FG_COLOR = "#1E1E1E"
GRID_COLOR = "#DEDEDE"


def fetch_data_local(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename, na_values=["NK", "N/K"])


@cache
def fetch_data_s3(key: str, bucket_name: str = S3_BUCKET) -> pd.DataFrame:
    """Fetches data from a S3 bucket and reads into a dataframe"""

    with tempfile.NamedTemporaryFile() as tmp:
        S3.Object(bucket_name, key).download_file(tmp.name)
        return fetch_data_local(tmp.name)


def get_age_bins(age: str) -> range:
    "Returns age bin sequence range from age string in format start-end"

    if age == "0":
        return range(0, 1)
    if "-" in age:
        start_age, end_age = list(map(int, age.split("-")))
    else:
        start_age = end_age = int(age)
    for i in range(len(AGE_BINS)):
        start_bin, end_bin = AGE_BINS[i]
        if start_bin <= start_age <= end_bin:
            start_index = i
        if start_bin <= end_age <= end_bin:
            end_index = i
    return range(start_index, end_index + 1)


def name_bin(bin_idx: int) -> str:
    bin = AGE_BINS[bin_idx]
    if bin[0] == bin[1]:
        return str(bin[0])
    if bin[0] == 80:
        return "80+"
    return f"{bin[0]}-{bin[1]}"


def get_age_bin_data(df: pd.DataFrame) -> pd.DataFrame:
    confirmed = df[df.Case_status == "confirmed"][["Age", "Gender"]]
    confirmed["Gender"] = confirmed.Gender.apply(
        lambda x: x.strip() if isinstance(x, str) else x
    )
    age_gender = (
        confirmed.groupby(["Age", "Gender"])
        .size()
        .reset_index()
        .rename(columns={0: "n"})
    )
    age_gender["Age_bins"] = age_gender.Age.map(get_age_bins)
    age_gender["distributed_n"] = age_gender.n / age_gender.Age_bins.map(len)

    data = []
    for row in age_gender.itertuples():
        for bin_idx in row.Age_bins:
            data.append((name_bin(bin_idx), row.Gender, row.distributed_n))
    final = pd.DataFrame(data, columns=["Bin", "Gender", "N"])
    return final.groupby(["Bin", "Gender"]).sum().reset_index()


def get_delays(
    df: pd.DataFrame, target_col: str, onset_col: str = "Date_onset"
) -> pd.Series:
    both = df[
        ~pd.isna(df[target_col])
        & ~pd.isna(df[onset_col])
        & df[target_col].astype(str).str.fullmatch(REGEX_DATE)
        & df[onset_col].astype(str).str.fullmatch(REGEX_DATE)
    ]
    try:
        both[target_col] = pd.to_datetime(both[target_col])
        both[onset_col] = pd.to_datetime(both[onset_col])
    except ParserError:
        logging.error("Error occured when parsing date from column")
        raise
    return both[target_col] - both[onset_col]


def get_epicurve(df: pd.DataFrame, cumulative: bool = True) -> pd.DataFrame:
    """Returns epidemic curve - number of cases by (estimated) date of symptom onset"""
    df["Date_onset_estimated"] = df.Date_onset_estimated.map(
        lambda x: pd.to_datetime(x)
        if isinstance(x, str) and re.match(REGEX_DATE, x)
        else None
    )
    grouped_by_onset = (
        df[
            ~pd.isna(df.Date_onset_estimated)
            & df.Case_status.isin(["confirmed", "probable"])
        ]
        .groupby("Date_onset_estimated")
        .size()
    )
    if not cumulative:
        return (
            grouped_by_onset.sum()
            .reset_index()
            .sort_values(by="Date_onset_estimated")
            .rename({0: "Num_cases"}, axis=1)
        )
    else:
        return (
            grouped_by_onset.cumsum()
            .reset_index()
            .sort_values(by="Date_onset_estimated")
            .rename({0: "Cumulative_cases"}, axis=1)
        )


def get_counts(df: pd.DataFrame) -> dict[str, int]:
    status = df.Case_status.value_counts()
    confirmed = df[df.Case_status == "confirmed"]
    return {
        "n_confirmed": int(status.confirmed),
        "n_probable": int(status.get("probable", 0)),
        "date": str(df[~pd.isna(df.Data_up_to)].Data_up_to.max()),
        "pc_valid_age_gender": percentage_occurrence(
            confirmed,
            (~confirmed.Age.isna()) & (~confirmed.Gender.isna()),
        ),
    }


def get_timeseries_location_status(
    df: pd.DataFrame, fill_index: bool = False
) -> pd.DataFrame:
    "Returns a time series case dataset (number of cases by location by date stratified by confirmed and probable)"
    statuses = ["confirmed", "probable"]
    df = df[
        df.Case_status.isin(statuses)
        & ~pd.isna(df.Date_onset_estimated)
        & ~pd.isna(df.Location_District)
    ]
    locations = sorted(set(df.Location_District)) + [None]
    mindate, maxdate = df.Date_onset_estimated.min(), df.Date_onset_estimated.max()

    def timeseries_for_location(location: str | None) -> pd.DataFrame:
        if location is None:
            counts = (
                df.groupby(["Date_onset_estimated", "Case_status"])
                .size()
                .reset_index()
                .pivot(index="Date_onset_estimated", columns="Case_status", values=0)
                .fillna(0)
                .astype(int)
            )
        else:
            counts = (
                df[df.Location_District == location]
                .groupby(["Date_onset_estimated", "Case_status"])
                .size()
                .reset_index()
                .pivot(index="Date_onset_estimated", columns="Case_status", values=0)
                .fillna(0)
                .astype(int)
            )
        if fill_index:
            counts = counts.reindex(pd.date_range(mindate, maxdate), fill_value=0)
        for status in set(statuses) - set(counts.columns):
            counts[status] = 0
        counts = counts.rename(columns={s: "daily_" + s for s in statuses})
        for s in statuses:
            counts["cumulative_" + s] = counts["daily_" + s].cumsum()
        counts["Location_District"] = location if location else "Total"
        return counts

    timeseries = pd.concat(map(timeseries_for_location, locations)).fillna(0)
    for col in ["daily_" + s for s in statuses] + ["cumulative_" + s for s in statuses]:
        timeseries[col] = timeseries[col].astype(int)
    return timeseries.reset_index(names="Date_onset_estimated")


def plot_timeseries_location_status(df: pd.DataFrame, columns: int = 3):
    df = get_timeseries_location_status(df, fill_index=True)
    locations = sorted(set(df.Location_District) - {"Total"})

    fig = make_subplots(
        rows=2, cols=3, subplot_titles=locations, shared_yaxes=True, shared_xaxes=True
    )

    for i, location in enumerate(locations):
        location_data = df[df.Location_District == location]
        cur_row, cur_col = i // columns + 1, i % columns + 1
        fig.add_trace(
            go.Scatter(
                x=location_data.Date_onset_estimated,
                y=location_data.cumulative_confirmed,
                name="confirmed",
                line_color=PRIMARY_COLOR,
                line_width=3,
                showlegend=not bool(i),
            ),
            row=cur_row,
            col=cur_col,
        )
        fig.add_trace(
            go.Scatter(
                x=location_data.Date_onset_estimated,
                y=location_data.cumulative_probable,
                name="probable",
                line_color=SECONDARY_COLOR,
                line_width=3,
                showlegend=not bool(i),
            ),
            row=cur_row,
            col=cur_col,
        )
    fig.update_yaxes(
        range=[
            0,
            max(
                df[df.Location_District != "Total"].cumulative_confirmed.max(),
                df[df.Location_District != "Total"].cumulative_probable.max(),
            )
            + 1,
        ],
        gridcolor=GRID_COLOR,
    )
    fig.update_xaxes(
        gridcolor=GRID_COLOR,
    )
    fig.update_layout(
        plot_bgcolor=BG_COLOR,
        font_family=FONT,
        paper_bgcolor=BG_COLOR,
        hoverlabel_font_family=FONT,
        legend_font_family=TITLE_FONT,
        legend_font_size=LEGEND_FONT_SIZE,
    )
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(
            family=TITLE_FONT, size=LEGEND_FONT_SIZE + 2, color=FG_COLOR
        )

    return fig


def render(template: Path, variables: dict[str, Any]) -> str:
    with template.open() as f:
        return chevron.render(f, variables)


def render_figure(fig, key: str) -> str:
    return {key: plotly.io.to_html(fig, include_plotlyjs=False, full_html=False)}


def plot_epicurve(df: pd.DataFrame, cumulative: bool = True):
    data = get_epicurve(df, cumulative=cumulative)
    fig = go.Figure()

    if cumulative:
        fig.add_trace(
            go.Scatter(
                x=data.Date_onset_estimated,
                y=data.Cumulative_cases,
                name="Cumulative cases",
                line_color=PRIMARY_COLOR,
                line_width=3,
            ),
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=data.Date_onset_estimated,
                y=data.Num_cases,
                name="Cases",
                line_color=PRIMARY_COLOR,
            ),
        )

    fig.update_xaxes(
        title_text="Date of symptom onset",
        title_font_family=TITLE_FONT,
        title_font_color=FG_COLOR,
        gridcolor=GRID_COLOR,
    )

    fig.update_yaxes(
        title_text="Cumulative cases" if cumulative else "Cases",
        title_font_family=TITLE_FONT,
        title_font_color=FG_COLOR,
        gridcolor=GRID_COLOR,
    )
    fig.update_layout(
        plot_bgcolor=BG_COLOR,
        font_family=FONT,
        paper_bgcolor=BG_COLOR,
        hoverlabel_font_family=FONT,
    )
    return fig


def percentage_occurrence(df: pd.DataFrame, filter_series: pd.Series) -> int:
    """Returns percentage occurrence of filter_series within a dataframe"""
    return int(round(100 * sum(filter_series) / len(df)))


def plot_delay_distribution(
    df: pd.DataFrame,
    col: str,
    title: str,
    index: str,
    max_delay_days: int = 30,
):
    delays = get_delays(df, col).dt.days.value_counts()
    if max_delay_days not in delays:
        delays[max_delay_days] = 0
    delays = delays.reset_index().rename({"index": title, 0: "count"}, axis=1)
    fig = px.bar(
        delays,
        x=title,
        y="count",
        color_discrete_sequence=[PRIMARY_COLOR],
    )
    fig.update_layout(
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        font_family=FONT,
        title=index,
        title_font_color=FG_COLOR,
        title_font_family=TITLE_FONT,
        hoverlabel_font_family=FONT,
        bargap=0.2,
    )
    fig.update_xaxes(
        title_font_family=TITLE_FONT,
        gridcolor=GRID_COLOR,
        linecolor=GRID_COLOR,
        title_font_color=FG_COLOR,
        linewidth=3,
    )
    fig.update_yaxes(
        title_font_family=TITLE_FONT,
        title_font_color=FG_COLOR,
        gridcolor=GRID_COLOR,
        showline=False,
    )

    return fig


def plot_age_gender(df: pd.DataFrame):
    df = get_age_bin_data(df)
    fig = go.Figure()
    vals = {}
    for row in df.itertuples():
        vals[(row.Bin, row.Gender)] = row.N

    bin_names = [name_bin(bin_idx) for bin_idx in range(len(AGE_BINS))]
    female_binvals = -np.array([vals.get((bin, "female"), 0) for bin in bin_names])
    male_binvals = np.array([vals.get((bin, "male"), 0) for bin in bin_names])

    y = bin_names
    max_binval = max(-female_binvals.max(), male_binvals.max())
    nearest = int(((max_binval // 5) + 1) * 5)
    ticks = np.linspace(-nearest, nearest, 2 * nearest + 1).astype(int)

    fig.update_yaxes(title="Age", title_font_family=TITLE_FONT, gridcolor=GRID_COLOR)
    fig.update_xaxes(
        range=[-nearest, nearest],
        tickvals=ticks,
        ticktext=list(map(abs, ticks)),
        title="Counts",
        title_font_family=TITLE_FONT,
        gridcolor=GRID_COLOR,
        title_font_color=FG_COLOR,
        zeroline=False,
    )
    fig.update_layout(
        dict(
            barmode="overlay",
            bargap=0.1,
            template="plotly_white",
            font_family=FONT,
            hoverlabel_font_family=FONT,
            plot_bgcolor=BG_COLOR,
            paper_bgcolor=BG_COLOR,
            legend_font_family=TITLE_FONT,
            legend_font_size=LEGEND_FONT_SIZE,
        )
    )

    fig.add_trace(
        go.Bar(
            y=y,
            x=male_binvals,
            orientation="h",
            name="male",
            hoverinfo="skip",
            hovertemplate=None,
            textposition="none",
            marker=dict(color=SECONDARY_COLOR),
        )
    )
    fig.add_trace(
        go.Bar(
            y=y,
            x=female_binvals,
            orientation="h",
            name="female",
            text=-female_binvals.astype("int"),
            hoverinfo="skip",
            hovertemplate=None,
            textposition="none",
            marker=dict(color=BLUE_PRIMARY_COLOR),
        )
    )

    return fig


def store_s3(
    data: str,
    key: str | list[str],
    bucket_name: str,
    content_type: str,
):
    keys = [key] if isinstance(key, str) else key
    if not os.getenv("SKIP_UPLOAD"):
        for k in keys:
            logging.info(f"Uploading data to s3://{bucket_name}/{k}")
            try:
                S3.Object(bucket_name, k).put(Body=data, ContentType=content_type)
            except Exception:
                logging.exception("An exception occurred while trying to upload files")
                raise


def invalidate_cache(
    distribution_id: str,
    paths: list[str] = ["/", "/index.html", "/metadata.json"],
):
    "Invalidates CloudFront cache"
    try:
        invalidation = boto3.client("cloudfront").create_invalidation(
            DistributionId=distribution_id,
            InvalidationBatch={
                "Paths": {"Quantity": len(paths), "Items": paths},
                "CallerReference": f"marburg_report_{datetime.datetime.now().isoformat()}",
            },
        )
        logging.info(f"Invalidation ID: {invalidation['Invalidation']['Id']}")
    except Exception:
        logging.info("Exception occurred when trying to invalidate existing cache")
        raise


def build(
    fetch_bucket: str,
    date: datetime.date = None,
    overrides=OVERRIDES,
):
    """Build Marburg 2023 epidemiological report for a particular date"""
    date = date or datetime.datetime.today().date()
    var = {"published_date": str(date)}

    if date in overrides:
        overrides = overrides[date]
        logging.info(f"Found overrides for {date}")

    try:
        df = fetch_data_s3("latest.csv")
    except ValueError as e:
        logging.error(e)
        sys.exit(1)
    except ConnectionError as e:
        logging.error(e)
        sys.exit(1)

    var.update(get_counts(df))
    var.update(render_figure(plot_epicurve(df), "embed_epicurve"))
    var.update(
        render_figure(
            plot_timeseries_location_status(df), "embed_epicurve_location_status"
        )
    )
    var.update(render_figure(plot_age_gender(df), "embed_age_gender"))
    var.update(
        render_figure(
            plot_delay_distribution(
                df, "Date_of_first_consult", "Delay to consultation from onset", "A", 20
            ),
            "embed_delay_distribution_consult",
        )
    )
    var.update(
        render_figure(
            plot_delay_distribution(
                df, "Date_death", "Delay to death from onset", "B", 20
            ),
            "embed_delay_distribution_death",
        )
    )

    logging.info("Writing report")

    report_data = render(
        Path("index_template.html"),
        var,
    )

    logging.info("Writing metadata")

    metadata = json.dumps(
        {k: v for k, v in var.items() if not k.startswith("embed_")},
        indent=2,
        sort_keys=True,
    )

    # write locally to enable preview
    Path("report.html").write_text(report_data)

    store_s3(
        report_data,
        ["index.html", f"{date}.html"],
        bucket_name=S3_BUCKET_REPORT,
        content_type="text/html",
    )
    store_s3(
        metadata,
        "metadata.json",
        bucket_name=S3_BUCKET_REPORT,
        content_type="application/json",
    )
    if distribution_id := os.getenv("CLOUDFRONT_DISTRIBUTION"):
        invalidate_cache(distribution_id)

    # save timeseries to aggregates
    timeseries_location_status = get_timeseries_location_status(df)
    store_s3(
        timeseries_location_status.to_csv(index=False),
        [
            "timeseries-location-status/latest.csv",
            f"timeseries-location-status/{date}.csv",
        ],
        bucket_name=S3_AGGREGATES,
        content_type="text/csv",
    )


if __name__ == "__main__":
    print(
        """
Marburg 2023 Epidemiology Report generator

By default, this will read files from S3_BUCKET and generate a report for
the current date, which can be overridden in the DATE env var.

The generated report will be stored in S3_BUCKET_REPORT and a CloudFront
invalidation performed to refresh the report. Set SKIP_UPLOAD=y to skip
report upload, this can be used for local report generation"""
    )
    DATE = os.getenv("DATE")
    build(
        S3_BUCKET, date=datetime.datetime.fromisoformat(DATE).date() if DATE else None
    )
