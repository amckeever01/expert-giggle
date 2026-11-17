"""Streamlit dashboard for Benford analysis and two-period comparisons."""

from __future__ import annotations

import io
import math
import zipfile
from decimal import Decimal, InvalidOperation

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import chisquare


MIN_EXPECTED_COUNT = 5


def _leading_digits(value: object, count: int) -> int | None:
    """Return the first one or two significant digits of a positive number."""
    try:
        number = Decimal(str(value)).copy_abs().normalize()
    except (InvalidOperation, ValueError):
        return None

    if not number.is_finite() or number.is_zero():
        return None

    parts = number.as_tuple()
    digits = "".join(str(digit) for digit in parts.digits)
    if parts.exponent > 0:
        digits += "0" * parts.exponent
    if len(digits) < count:
        return None
    return int(digits[:count])


def _benford_analysis(values: pd.Series, digit_count: int) -> pd.DataFrame:
    """Build observed and expected Benford frequencies for positive values."""
    lower = 1 if digit_count == 1 else 10
    upper = 9 if digit_count == 1 else 99
    digits = np.arange(lower, upper + 1)
    valid_values = pd.to_numeric(values, errors="coerce")
    valid_values = valid_values[np.isfinite(valid_values) & (valid_values > 0)]
    leading = valid_values.map(lambda value: _leading_digits(value, digit_count))
    leading = leading.dropna().astype(int)
    expected = np.log10(1 + 1 / digits)
    counts = leading.value_counts().reindex(digits, fill_value=0)

    return pd.DataFrame(
        {
            "Digits": digits,
            "Observed count": counts.to_numpy(),
            "Actual proportion": counts.to_numpy() / max(len(leading), 1),
            "Benford expected": expected,
            "Expected count": expected * len(leading),
        }
    )


def _chi_square_result(distribution: pd.DataFrame) -> dict[str, object]:
    """Compare observed and Benford counts, pooling bins with small expectations."""
    observed = distribution["Observed count"].to_numpy(dtype=float)
    expected = distribution["Expected count"].to_numpy(dtype=float)
    groups: list[tuple[int, int, float, float]] = []
    start = 0
    observed_total = expected_total = 0.0

    for index, (actual, theoretical) in enumerate(zip(observed, expected)):
        observed_total += actual
        expected_total += theoretical
        if expected_total >= MIN_EXPECTED_COUNT:
            groups.append((start, index, observed_total, expected_total))
            start = index + 1
            observed_total = expected_total = 0.0

    if expected_total > 0:
        if groups:
            prior_start, _, prior_observed, prior_expected = groups[-1]
            groups[-1] = (
                prior_start,
                len(expected) - 1,
                prior_observed + observed_total,
                prior_expected + expected_total,
            )
        else:
            groups.append((0, len(expected) - 1, observed_total, expected_total))

    if len(groups) < 2:
        return {
            "Statistic": None,
            "p-value": None,
            "Degrees of freedom": 0,
            "Bins tested": len(groups),
            "Conclusion": (
                "Not enough data for a chi-square test with at least two "
                "bins having adequate expected counts."
            ),
        }

    grouped_observed = np.array([group[2] for group in groups])
    grouped_expected = np.array([group[3] for group in groups])
    statistic, p_value = chisquare(grouped_observed, f_exp=grouped_expected)
    return {
        "Statistic": float(statistic),
        "p-value": float(p_value),
        "Degrees of freedom": len(groups) - 1,
        "Bins tested": len(groups),
        "Conclusion": (
            "Significant difference from Benford distribution (p < 0.05)."
            if p_value < 0.05
            else "No statistically significant difference detected (p >= 0.05)."
        ),
    }


def _ks_two_sample(first: pd.Series, second: pd.Series) -> dict[str, object]:
    """Compute the two-sample KS statistic and its asymptotic p-value with NumPy."""
    first_values = pd.to_numeric(first, errors="coerce").to_numpy(dtype=float)
    second_values = pd.to_numeric(second, errors="coerce").to_numpy(dtype=float)
    first_values = np.sort(first_values[np.isfinite(first_values)])
    second_values = np.sort(second_values[np.isfinite(second_values)])

    if len(first_values) == 0 or len(second_values) == 0:
        return {
            "Statistic": None,
            "p-value (approx.)": None,
            "Sample 1 size": len(first_values),
            "Sample 2 size": len(second_values),
            "Conclusion": "Both periods need at least one numeric observation.",
        }

    pooled = np.sort(np.concatenate((first_values, second_values)))
    first_cdf = np.searchsorted(first_values, pooled, side="right") / len(first_values)
    second_cdf = np.searchsorted(second_values, pooled, side="right") / len(second_values)
    statistic = float(np.max(np.abs(first_cdf - second_cdf)))
    effective_n = math.sqrt(
        len(first_values) * len(second_values)
        / (len(first_values) + len(second_values))
    )
    scaled_statistic = (effective_n + 0.12 + 0.11 / effective_n) * statistic
    p_value = 1.0 if scaled_statistic == 0 else 0.0
    if scaled_statistic > 0:
        for term in range(1, 101):
            contribution = 2 * (-1) ** (term - 1) * math.exp(
                -2 * term**2 * scaled_statistic**2
            )
            p_value += contribution
            if abs(contribution) < 1e-12:
                break
    p_value = float(np.clip(p_value, 0, 1))

    return {
        "Statistic": statistic,
        "p-value (approx.)": p_value,
        "Sample 1 size": len(first_values),
        "Sample 2 size": len(second_values),
        "Conclusion": (
            "The period distributions differ significantly (p < 0.05)."
            if p_value < 0.05
            else "No statistically significant difference detected (p >= 0.05)."
        ),
    }


def _read_upload(uploaded_file: object, extension: str) -> pd.DataFrame:
    if extension == "csv":
        return pd.read_csv(uploaded_file)
    workbook = pd.ExcelFile(uploaded_file)
    if len(workbook.sheet_names) == 1:
        return pd.read_excel(workbook, sheet_name=workbook.sheet_names[0])
    sheet = st.sidebar.selectbox("Excel worksheet", workbook.sheet_names)
    return pd.read_excel(workbook, sheet_name=sheet)


def _parse_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_convert(None)


def _apply_filters(data: pd.DataFrame) -> pd.DataFrame:
    filtered = data.copy()
    columns = [None, *data.columns]
    with st.sidebar.expander("Filter data (up to three criteria)", expanded=True):
        for index in range(3):
            column = st.selectbox(
                f"Filter {index + 1}: column",
                columns,
                format_func=lambda value: "(No filter)" if value is None else str(value),
                key=f"filter_column_{index}",
            )
            if column is None:
                continue

            series = filtered[column]
            if pd.api.types.is_numeric_dtype(series):
                numeric = pd.to_numeric(series, errors="coerce")
                finite = numeric[np.isfinite(numeric)]
                if finite.empty:
                    st.caption(f"{column} has no finite numeric values to filter.")
                    continue
                minimum, maximum = float(finite.min()), float(finite.max())
                if minimum == maximum:
                    filtered = filtered[numeric == minimum]
                    st.caption(f"{column} = {minimum:g}")
                else:
                    step = 1.0 if pd.api.types.is_integer_dtype(series) else (maximum - minimum) / 100
                    selected = st.slider(
                        f"{column} range",
                        min_value=minimum,
                        max_value=maximum,
                        value=(minimum, maximum),
                        step=step,
                        key=f"filter_range_{index}",
                    )
                    filtered = filtered[numeric.between(*selected)]
            elif pd.api.types.is_datetime64_any_dtype(series):
                parsed = _parse_dates(series)
                dates = parsed.dropna().dt.date
                if dates.empty:
                    st.caption(f"{column} has no valid dates to filter.")
                    continue
                selected = st.date_input(
                    f"{column} range",
                    value=(dates.min(), dates.max()),
                    key=f"filter_date_{index}",
                )
                if isinstance(selected, tuple) and len(selected) == 2:
                    filtered = filtered[
                        parsed.dt.date.between(selected[0], selected[1]).fillna(False)
                    ]
            else:
                options = series.dropna().drop_duplicates().tolist()
                selected_values = st.multiselect(
                    f"{column} values (leave empty for all)",
                    options,
                    format_func=str,
                    key=f"filter_values_{index}",
                )
                if selected_values:
                    filtered = filtered[series.isin(selected_values)]
    return filtered


def _download_bundle(
    filtered: pd.DataFrame,
    summary: pd.DataFrame,
    benford: pd.DataFrame,
    tests: pd.DataFrame,
    report: str,
) -> bytes:
    files = {
        "filtered_data.csv": filtered.to_csv(index=False).encode("utf-8"),
        "summary_statistics.csv": summary.to_csv(index=False).encode("utf-8"),
        "benford_analysis.csv": benford.to_csv(index=False).encode("utf-8"),
        "statistical_tests.csv": tests.to_csv(index=False).encode("utf-8"),
        "analysis_report.md": report.encode("utf-8"),
    }
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, content in files.items():
            archive.writestr(filename, content)
    return bundle.getvalue()


def main() -> None:
    st.set_page_config(page_title="Benford & Time-Period Analyzer", layout="wide")
    st.title("Benford & Time-Period Analyzer")
    st.write(
        "Upload a CSV or Excel file to explore Benford's Law, filter records, "
        "and compare numeric values across two time periods."
    )
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
    if uploaded_file is None:
        st.info("Upload a file to begin. Data is analyzed in this app session.")
        return

    extension = uploaded_file.name.rsplit(".", 1)[-1].lower()
    try:
        data = _read_upload(uploaded_file, extension)
    except Exception as error:
        st.error(f"Could not read the uploaded file: {error}")
        return
    if data.empty or len(data.columns) == 0:
        st.error("The uploaded file has no rows or columns to analyze.")
        return

    filtered = _apply_filters(data)
    numeric_columns = [
        column for column in filtered.columns
        if pd.api.types.is_numeric_dtype(filtered[column])
    ]
    if not numeric_columns:
        st.error("The filtered data has no numeric columns to analyze.")
        return

    st.subheader("Filtered data")
    st.caption(f"{len(filtered):,} of {len(data):,} rows")
    st.dataframe(filtered.head(100), width="stretch")
    numeric_column = st.selectbox("Numeric column to analyze", numeric_columns)
    numeric_values = pd.to_numeric(filtered[numeric_column], errors="coerce")

    st.subheader("Summary statistics")
    summary_series = numeric_values.describe()
    summary = summary_series.rename_axis("Statistic").reset_index(name="Value")
    st.dataframe(summary, width="stretch", hide_index=True)

    st.subheader("Benford analysis")
    st.caption(
        "Goodness-of-fit uses a chi-square test. Adjacent digit bins are pooled "
        "until each tested bin has at least five expected observations."
    )
    benford_tables: list[pd.DataFrame] = []
    test_rows: list[dict[str, object]] = []
    for digit_count in (1, 2):
        distribution = _benford_analysis(numeric_values, digit_count)
        test = _chi_square_result(distribution)
        label = "First digit" if digit_count == 1 else "First two digits"
        distribution.insert(0, "Analysis", label)
        benford_tables.append(distribution)
        test_rows.append({"Analysis": label, **test})
        st.markdown(f"**{label}** — {int(distribution['Observed count'].sum()):,} eligible values")
        st.line_chart(
            distribution.set_index("Digits")[["Actual proportion", "Benford expected"]],
            height=250,
        )
        st.dataframe(pd.DataFrame([test]), width="stretch", hide_index=True)

    st.subheader("Compare two time periods (two-sample KS test)")
    date_column = st.selectbox("Date/time column", filtered.columns)
    parsed_dates = _parse_dates(filtered[date_column])
    valid_dates = parsed_dates.dropna()
    ks_result: dict[str, object] | None = None
    if valid_dates.empty:
        st.warning(f"{date_column} has no values that could be parsed as dates.")
    else:
        available_dates = sorted(valid_dates.dt.date.unique())
        min_date, max_date = available_dates[0], available_dates[-1]
        midpoint = len(available_dates) // 2
        if midpoint > 0:
            first_default = (min_date, available_dates[midpoint - 1])
            second_default = (available_dates[midpoint], max_date)
        else:
            first_default = second_default = (min_date, max_date)
        period_one, period_two = st.columns(2)
        with period_one:
            first_range = st.date_input(
                "Period 1",
                min_value=min_date,
                max_value=max_date,
                value=first_default,
                key="ks_period_one",
            )
        with period_two:
            second_range = st.date_input(
                "Period 2",
                min_value=min_date,
                max_value=max_date,
                value=second_default,
                key="ks_period_two",
            )

        if (
            isinstance(first_range, tuple)
            and len(first_range) == 2
            and isinstance(second_range, tuple)
            and len(second_range) == 2
        ):
            date_only = parsed_dates.dt.date
            first_mask = date_only.between(*first_range).fillna(False)
            second_mask = date_only.between(*second_range).fillna(False)
            ks_result = _ks_two_sample(
                numeric_values[first_mask],
                numeric_values[second_mask],
            )
            st.caption("The KS p-value is an asymptotic approximation; significance level α = 0.05.")
            st.dataframe(pd.DataFrame([ks_result]), width="stretch", hide_index=True)
            if ks_result["Statistic"] is not None:
                first_values = np.sort(
                    numeric_values[first_mask].dropna().to_numpy(dtype=float)
                )
                second_values = np.sort(
                    numeric_values[second_mask].dropna().to_numpy(dtype=float)
                )
                pooled_values = np.sort(np.unique(np.concatenate((first_values, second_values))))
                ecdf = pd.DataFrame(
                    {
                        "Period 1": np.searchsorted(
                            first_values, pooled_values, side="right"
                        ) / len(first_values),
                        "Period 2": np.searchsorted(
                            second_values, pooled_values, side="right"
                        ) / len(second_values),
                    },
                    index=pooled_values,
                )
                ecdf.index.name = "Value"
                st.line_chart(ecdf, height=250)

    tests = pd.DataFrame(test_rows)
    if ks_result is not None:
        tests = pd.concat(
            [tests, pd.DataFrame([{"Analysis": "Two-sample KS test", **ks_result}])],
            ignore_index=True,
        )
    benford = pd.concat(benford_tables, ignore_index=True)
    report_lines = [
        "# Benford & Time-Period Analysis",
        "",
        f"- Uploaded file: {uploaded_file.name}",
        f"- Rows after filters: {len(filtered)} of {len(data)}",
        f"- Numeric column: {numeric_column}",
        "",
        "## Summary statistics",
        summary.to_markdown(index=False),
        "",
        "## Statistical tests",
        tests.to_markdown(index=False),
        "",
        "Visualizations are represented by the downloadable Benford and filtered-data CSV files.",
    ]
    report = "\n".join(report_lines)
    st.subheader("Download and share results")
    st.caption(
        "The ZIP includes filtered data, summary statistics, Benford chart data, "
        "test results, and a Markdown report."
    )
    st.download_button(
        "Download analysis bundle (ZIP)",
        data=_download_bundle(filtered, summary, benford, tests, report),
        file_name="benford_analysis_bundle.zip",
        mime="application/zip",
    )
    st.download_button(
        "Download report (Markdown)",
        data=report,
        file_name="analysis_report.md",
        mime="text/markdown",
    )


if __name__ == "__main__":
    main()
