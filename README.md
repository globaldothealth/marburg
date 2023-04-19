# Marburg Case Data Equatorial Guinea outbreak 2023

This repository contains dated records of curated Marburg case data from the 2023 outbreak in Equatorial Guinea. Data are curated from openly accessible sources only. As new data become available we are updating the line list.

**[Latest data and archives](https://l66noa47nk.execute-api.eu-central-1.amazonaws.com/web)** | **[Summary report](https://www.marburg.global.health)**

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

UPDATE
2023-04-19: Confirmed case counts are updated from MINSABS and WHO reporting. Location data for the first 15 confirmed cases is consistent with MINSABS reporting through the [Epidemiological update on 2023-04-16]( https://www.guineasalud.org/archivos/Informes/Informe16042023.pdf) [n=15. Bata 9; Ebibeyin 3; Evinayong 2; Nsork 1]. Confirmed case number 16 was added to our line-list using data from the [WHO Director-General media briefing](https://www.who.int/director-general/speeches/detail/who-director-general-s-opening-remarks-at-media-briefing-18-april-2023).

Probable cases (n=20) are included in our line-list using data from [MINSABS Official Statement No. 3, Page 7]( https://www.guineasalud.org/archivos/Ordenes/Comunicado3.pdf). WHO reported an update on [2023-04-15](https://www.who.int/emergencies/disease-outbreak-news/item/2023-DON459) increasing the probable count to 23. Global.health is working to reconcile our case and location data; however, due to the presentation of aggregated data in official reporting, we may be unable to update or continue to track probable case data in our line-list moving forward. 

Three suspected cases, now discarded, are included in our line-list using data from [MINSABS Official Statement No. 3, Page 7]( https://www.guineasalud.org/archivos/Ordenes/Comunicado3.pdf). Suspected cases are not included in our line-list after this report.


This section is an overview of the data curation process, a discussion about limitations and assumptions.

The Marburg line-list is built by checking a collection of sources, listed here, which will be updated as new sources become available: https://github.com/globaldothealth/marburg/wiki. The original source(s) of information is provided for each line-list ID in our database. Data released from Ministerio de Sanidad y Bienestar Social de la República de Guinea Ecuatorial (MINSABS) has been our primary source of information (<https://guineasalud.org/>). Our line-list also includes publicly available data from the World Health Organization.

Metadata are added at any time, as information becomes available and our time and resources permit. After making changes, the case will be recorded as modified with the date. Multiple curators look at each datapoint and any discrepancies are resolved in conversations between them. We remain limited by inconsistent, aggregated, or missing case information; change in reporting format; data reconciliation; reporting delays; and change in case definitions, among other reasons. Assumptions are made that may compromise the accuracy of the data.

Users should refer to our [data dictionary](data_dictionary.yml) for a description of each variable. Limitations and assumptions for select variables are briefly discussed below.

**Case_status**: Only confirmed and probable cases are logged at this time. 

**Date_onset**: Information is available for a selection of cases.

**Outcome. Type: Death**: The report date is used when a Date_death is not specified by source. 

**Outcome. Type: Recovered**: The report date is used when a Date_recovered is not specified by source. 

**Healthcare_worker**: Due to the limited availability of information, we have not been able to log every HCW case or outcome.

Data are hand-curated. The process and methods to create, organize, and maintain data have been applied with consistency; however, we’re human and mistakes happen. As stated above, line-list data may change due to ongoing data reconciliation and validation. We welcome your contributions and feedback. Get involved!

## Contributing

If you would like to request changes, [open an issue](https://github.com/globaldothealth/marburg/issues/new) on this repository and we will happily consider your request.

If requesting a fix please include steps to reproduce undesirable behaviors.

If you would like to contribute, assign an issue to yourself and/or reach out to a contributor and we will happily help you help us.

## License and attribution

This repository is published under MIT License and data exports are published under the CC BY 4.0 license.

Please cite as: "Global.health Marburg (accessed on YYYY-MM-DD)" & please add the appropriate agency, paper, and/or individual in publications and/or derivatives using these data, contact them regarding the legal use of these data, and remember to pass-forward any existing license/warranty/copyright information.

Please also refer to the original source of the data: Ministerio de Sanidad y Bienestar Social de la República de Guinea Equatorial (<https://guineasalud.org/>).
