# Marburg Case Data Equatorial Guinea outbreak 2023

This repository contains dated records of curated Marburg case data from the 2023 outbreak in Equatorial Guinea. Data are curated from openly accessible sources only.

**[Latest data and archives](https://l66noa47nk.execute-api.eu-central-1.amazonaws.com/web)** | **[Summary report](https://www.marburg.global.health)**

Our first line list was published on March 30th, 2023 and is based on data up to March 21st, 2023. As new data become available we are updating the line list. The line list is created directly from a report from the Ministerio de Sanidad y Bienestar Social de la República de Guinea Equatorial (<https://guineasalud.org/>).

There may be discrepancies and data remains limited at this stage of the outbreak. Should you find additional detail or have comments about the accuracy of information supplied here please address questions to info@global.health.

## Getting the data

There are two datasets: a **line list** and a **timeseries aggregation**. Only cases that have either a Date_onset or where Date_onset can be estimated (from Date_death or Date_of_first_consult) are visible in the timeseries.

**Python**

```python
import pandas as pd
df = pd.read_csv("https://l66noa47nk.execute-api.eu-central-1.amazonaws.com/web/url?folder=&file_name=latest.csv")

# aggregate timeseries data by location and status
ts = pd.read_csv("https://marburg-aggregates.s3.eu-central-1.amazonaws.com/timeseries-location-status/latest.csv")
```

**R**

```R
df <- read.csv("https://l66noa47nk.execute-api.eu-central-1.amazonaws.com/web/url?folder=&file_name=latest.csv")

# aggregate timeseries data by location and status
ts <- read.csv("https://marburg-aggregates.s3.eu-central-1.amazonaws.com/timeseries-location-status/latest.csv")
```

## Data curation

This section is an overview of the data curation process, a discussion about limitations and assumptions.

The Marburg Eq. Guinea line-list is built at this stage from data provided by the Ministerio de Sanidad y Bienestar Social de la República de Guinea Ecuatorial (<https://guineasalud.org/>). As we add additional sources we will list our sources in a separate file. The original source(s) of information is provided for each line-list ID in our database.

Users should refer to our [data dictionary](data_dictionary.yml) for a description of each variable. Assumptions for select variables are briefly discussed below.

**Case_status**: Only confirmed, probable, and suspected cases are logged at this time. We also list the number of contacts but no detail on these contacts is available.

**Date_onset**: Information is available for a selection of cases.

Data are hand-curated. The process and methods to create, organize, and maintain data have been applied with consistency. As stated above, line-list data may change due to ongoing data reconciliation and validation. We welcome your contributions and feedback.

## Contributing

If you would like to request changes, [open an issue](https://github.com/globaldothealth/marburg/issues/new) on this repository and we will happily consider your request.

If requesting a fix please include steps to reproduce undesirable behaviors.

If you would like to contribute, assign an issue to yourself and/or reach out to a contributor and we will happily help you help us.

## License and attribution

This repository is published under MIT License and data exports are published under the CC BY 4.0 license.

Please cite as: "Global.health Marburg (accessed on YYYY-MM-DD)" & please add the appropriate agency, paper, and/or individual in publications and/or derivatives using these data, contact them regarding the legal use of these data, and remember to pass-forward any existing license/warranty/copyright information.

Please also refer to the original source of the data: Ministerio de Sanidad y Bienestar Social de la República de Guinea Equatorial (<https://guineasalud.org/>).
