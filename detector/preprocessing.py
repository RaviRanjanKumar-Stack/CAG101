
import re
import pandas as pd

    
DATE_KEYWORDS = (
    "date",
    "posted",
    "posting",
    "timestamp",
    "txn_time",
    "value_date",
    "booking_date",
    "transaction_date",
    "voucher_date",
    "payment_date",
    "entry_date",
    "created_at",
    "updated_at",
)


ID_KEYWORDS = (
    "voucher", "identifier", "reference", "ref_no",
    "voucher_no", "account_code", "uuid", "guid",
)

AMOUNT_KEYWORDS = (
    "amount", "amt", "expenditure", "payment",
    "receipt", "debit", "credit", "balance",
    "total", "price", "rate", "value", "sum", "net",
)

BOOL_KEYWORDS = (
    "is_", "has_", "flag", "active", "enabled",
    "deleted", "valid",
)

CURRENCY_SYMBOLS = ("₹", "$", "€", "£", "¥", "INR", "USD")

TRUE_VALUES = {"true", "yes", "y", "t", "on", "active"}
FALSE_VALUES = {"false", "no", "n", "f", "off", "inactive"}

NUMERIC_THRESHOLD = 0.80
DATE_THRESHOLD = 0.80
BOOL_THRESHOLD = 0.95


def _read_csv_robust(uploaded_file):
    """Try common encodings when reading a CSV."""

    last_error = None

    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            uploaded_file.seek(0)

            df = pd.read_csv(
                uploaded_file,
                encoding=encoding,
                dtype=str,
                keep_default_na=True,
                skip_blank_lines=True,
                on_bad_lines="error",
            )

            return df, encoding

        except UnicodeDecodeError as exc:
            last_error = exc

        except pd.errors.EmptyDataError:
            raise ValueError("The CSV file is empty.")

        except pd.errors.ParserError:
            raise ValueError(
                "The CSV format is invalid. "
                "Please check its rows and separators."
            )

    raise ValueError(
        f"Could not decode the CSV file: {last_error}"
    )


def _clean_numeric_series(series):
    """Convert amount-like strings to numbers."""

    values = series.astype("string").str.strip()

    # Handle accounting-style negative values: (1,200.50)
    negative = values.str.match(
        r"^\s*\(.*\)\s*$",
        na=False,
    )

    values = values.str.replace(
        r"^\s*\(|\)\s*$",
        "",
        regex=True,
    )

    for symbol in CURRENCY_SYMBOLS:
        values = values.str.replace(
            symbol, "", regex=False
        )

    values = (
        values
        .str.replace(",", "", regex=False)
        .str.replace("\u00a0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("_", "", regex=False)
    )

    # Percentages need separate interpretation, so don't
    # silently convert them into ordinary monetary amounts.
    has_percent = values.str.contains("%", regex=False, na=False)
    values = values.str.replace("%", "", regex=False)

    result = pd.to_numeric(values, errors="coerce")

    result.loc[negative] = -result.loc[negative].abs()
    result.loc[has_percent] = float("nan")

    return result

#Pura Parse function likhta hu.

def _parse_dates(series):
    """Parse ISO dates first, then other dates using Indian day-first order."""
    values = series.astype("string").str.strip()

    # First parse standard ISO dates such as 2026-07-01
    iso_mask = values.str.match(
        r"^\d{4}-\d{2}-\d{2}(?:[ T].*)?$",
        na=False,
    )

    parsed = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns]",
    )

    parsed.loc[iso_mask] = pd.to_datetime(
        values.loc[iso_mask],
        errors="coerce",
        yearfirst=True,
    )

    # Parse remaining formats using Indian day-first convention
    other_mask = ~iso_mask
    parsed.loc[other_mask] = pd.to_datetime(
        values.loc[other_mask],
        errors="coerce",
        dayfirst=True,
    )

    return parsed



def _count_outliers(series):
    """Count statistical IQR outliers; not anomaly scores."""

    values = series.dropna()

    if len(values) < 4:
        return 0

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    if iqr == 0:
        return 0

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return int(
        ((values < lower) | (values > upper)).sum()
    )


def inspect_voucher_csv(uploaded_file):
    """
    Inspect and profile a voucher CSV.

    This function does not run the CAG 101
    graph-wavelet anomaly detection algorithm.
    """

    # 1. Read the complete CSV
    df, encoding_used = _read_csv_robust(uploaded_file)

    if df.empty:
        raise ValueError("The CSV contains no transactions.")

    if df.shape[1] == 0:
        raise ValueError("The CSV has no columns.")

    # 2. Preserve original column names
    original_columns = df.columns.tolist()

    # Normalise names for internal processing
    cleaned_columns = [
        re.sub(
            r"\s+",
            "_",
            str(column).strip().lower(),
        )
        for column in original_columns
    ]

    if any(not column for column in cleaned_columns):
        raise ValueError("A column has an empty name.")

    if len(set(cleaned_columns)) != len(cleaned_columns):
        raise ValueError(
            "Duplicate column names found after cleaning. "
            "Please make the column names unique."
        )

    df.columns = cleaned_columns

    # 3. Clean whitespace in text
    for column in df.columns:
        df[column] = df[column].map(
            lambda value: (
                value.strip()
                if isinstance(value, str)
                else value
            )
        )

        df[column] = df[column].replace("", pd.NA)

    # 4. Classify columns
    numeric_columns = []
    date_columns = []
    bool_columns = []
    id_columns = []
    categorical_columns = []
    text_columns = []
    schema = {}

    for column in df.columns:
        series = df[column]
        non_missing = series.notna().sum()
        name = column.lower()

        if non_missing == 0:
            schema[column] = {
                "type": "empty",
                "confidence": 0.0,
            }
            text_columns.append(column)
            continue

        # Identifier detection takes priority over numeric
        is_id = (
            any(word in name for word in ID_KEYWORDS)
            or name.endswith("_id")
            or name.endswith("_no")
            or name.endswith("_number")
            or name in {"id", "no", "number", "code"}
        )

        if is_id:
            id_columns.append(column)
            schema[column] = {
                "type": "identifier",
                "confidence": 1.0,
            }
            continue

        # 4a. Date columns: only parse when the name suggests a date
        looks_like_date = any(
            word in name for word in DATE_KEYWORDS
        ) or name.endswith("_dt")

        if looks_like_date:
            parsed_dates = _parse_dates(series)
            valid_ratio = (
                parsed_dates.notna().sum() / non_missing
            )

            if valid_ratio >= DATE_THRESHOLD:
                df[column] = parsed_dates
                date_columns.append(column)

                schema[column] = {
                    "type": "datetime",
                    "confidence": round(valid_ratio, 3),
                }
                continue

        # 4b. Boolean columns: require a Boolean-like name
        looks_boolean = any(
            word in name for word in BOOL_KEYWORDS
        )

        if looks_boolean:
            normalised = series.astype("string").str.lower()

            mapped = normalised.map(
                lambda value: (
                    True if value in TRUE_VALUES
                    else False if value in FALSE_VALUES
                    else None
                )
            )

            valid_ratio = mapped.notna().sum() / non_missing

            if valid_ratio >= BOOL_THRESHOLD:
                df[column] = pd.Series(
                    mapped,
                    index=df.index,
                    dtype="boolean",
                )

                bool_columns.append(column)
                schema[column] = {
                    "type": "boolean",
                    "confidence": round(valid_ratio, 3),
                }
                continue

        # 4c. Numeric columns: require a numeric-like name
        looks_numeric = any(
            word in name for word in AMOUNT_KEYWORDS
        )

        if looks_numeric:
            parsed_numbers = _clean_numeric_series(series)
            valid_ratio = (
                parsed_numbers.notna().sum() / non_missing
            )

            if valid_ratio >= NUMERIC_THRESHOLD:
                df[column] = parsed_numbers
                numeric_columns.append(column)

                schema[column] = {
                    "type": "numeric",
                    "confidence": round(valid_ratio, 3),
                }
                continue

        # 4d. Remaining fields: categorical or free text
        unique_count = series.nunique(dropna=True)
        unique_ratio = unique_count / max(non_missing, 1)

        if unique_count <= 50 and unique_ratio <= 0.20:
            df[column] = df[column].astype("category")
            categorical_columns.append(column)

            schema[column] = {
                "type": "categorical",
                "confidence": round(1 - unique_ratio, 3),
            }
        else:
            text_columns.append(column)

            schema[column] = {
                "type": "text",
                "confidence": 1.0,
            }

    # 5. Data-quality report
    missing_values = {
        str(column): int(count)
        for column, count in df.isna().sum().items()
    }

    missing_percent = {
        str(column): round(float(percent * 100), 2)
        for column, percent in df.isna().mean().items()
    }

    unique_counts = {
        str(column): int(count)
        for column, count in df.nunique(dropna=True).items()
    }

    duplicate_rows = int(df.duplicated().sum())

    # 6. Numeric statistics
    numeric_stats = {}

    for column in numeric_columns:
        values = df[column].dropna()

        if values.empty:
            continue

        numeric_stats[column] = {
            "count": int(values.count()),
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "outliers_iqr": _count_outliers(values),
        }

    # 7. Simple data-quality score
    completeness = 100 * (
        1 - df.isna().sum().sum() / max(df.size, 1)
    )

    duplicate_rate = duplicate_rows / max(len(df), 1)
    uniqueness = 100 * (1 - duplicate_rate)

    quality_score = round(
        0.7 * completeness + 0.3 * uniqueness,
        2,
    )

    # 8. Issues for display
    issues = []

    if duplicate_rows:
        issues.append(
            f"{duplicate_rows} fully duplicated rows found."
        )

    high_missing = [
        column
        for column, percentage in missing_percent.items()
        if percentage > 50
    ]

    if high_missing:
        issues.append(
            f"More than 50% values are missing in: {high_missing}"
        )

    # 9. Preview as rows of strings for the HTML table
    preview_df = df.head(5)

    preview = [
        [
            "" if pd.isna(value) else str(value)
            for value in row
        ]
        for row in preview_df.itertuples(
            index=False,
            name=None,
        )
    ]

    # 10. Return report
    return {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "original_columns": original_columns,
        "encoding_used": encoding_used,
        "schema": schema,
        "numeric_columns": numeric_columns,
        "date_columns": date_columns,
        "bool_columns": bool_columns,
        "id_columns": id_columns,
        "categorical_columns": categorical_columns,
        "text_columns": text_columns,
        "missing_values": missing_values,
        "missing_percent": missing_percent,
        "unique_counts": unique_counts,
        "numeric_stats": numeric_stats,
        "duplicate_rows": duplicate_rows,
        "quality_score": quality_score,
        "issues": issues,
        "preview": preview,
    }
