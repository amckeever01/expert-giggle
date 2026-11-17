# Benford & Time-Period Analyzer

A Streamlit dashboard for exploring CSV and Excel data. It supports three
interactive filters, first-digit and first-two-digit Benford analysis, and a
two-sample Kolmogorov-Smirnov (KS) comparison between user-selected time
periods. Results include charts, summary statistics, test results, and a
downloadable ZIP report bundle.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Upload a CSV, `.xlsx`, or `.xls` file. Choose up to three filters, a numeric
column, and a date/time column. The KS p-value is an asymptotic approximation;
the dashboard uses a significance level of 0.05. Benford goodness-of-fit uses
a chi-square test with adjacent digit bins pooled to meet the expected-count
threshold.

## Deploy and share

This repository is public on GitHub. To share a hosted dashboard, deploy it
with [Streamlit Community Cloud](https://share.streamlit.io/), select this
repository and `app.py`, and share the generated app URL. Users can also
download and share the analysis ZIP from the dashboard.