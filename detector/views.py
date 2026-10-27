import pandas as pd

from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from .preprocessing import inspect_voucher_csv
from .voucher_anomaly import detect_anomalies


MAX_UPLOAD_SIZE = 10 * 1024 * 1024


def home(request):
    """Display the main upload page."""
    return render(request, "detector/upload.html")


@require_http_methods(["GET", "POST"])
def upload_csv(request):
    """Inspect an uploaded voucher CSV and display anomaly results."""

    if request.method == "GET":
        return render(request, "detector/upload.html")

    # 1. Get the uploaded file.
    uploaded_file = request.FILES.get("csv_file")

    if uploaded_file is None:
        return render(
            request,
            "detector/upload.html",
            {"error": "Please select a CSV file first."},
        )

    # 2. Validate file type.
    if not uploaded_file.name.lower().endswith(".csv"):
        return render(
            request,
            "detector/upload.html",
            {"error": "Please upload a CSV file."},
        )

    # 3. Validate file size.
    if uploaded_file.size > MAX_UPLOAD_SIZE:
        return render(
            request,
            "detector/upload.html",
            {"error": "File is too large. Maximum size is 10 MB."},
        )

    try:
        # 4. Inspect the CSV structure and data quality.
        inspection = inspect_voucher_csv(uploaded_file)

        print("CSV INSPECTION SUCCESSFUL")
        print("Rows:", inspection.get("row_count"))

        # 5. Read the CSV for anomaly detection.
        uploaded_file.seek(0)

        encoding = inspection.get("encoding_used") or "utf-8-sig"

        df = pd.read_csv(uploaded_file, encoding=encoding)

        # Clean column names for reliable matching.
        df.columns = [str(col).strip() for col in df.columns]

        row_count = len(df)
        column_count = len(df.columns)

        # 6. Find suitable date and amount columns.
        date_candidates = [
            col for col in df.columns
            if "date" in col.lower() or "time" in col.lower()
        ]

        amount_candidates = [
            col for col in df.columns
            if any(
                word in col.lower()
                for word in (
                    "amount",
                    "expenditure",
                    "payment",
                    "debit",
                    "credit",
                    "value",
                    "total",
                )
            )
        ]

        print("CSV HEADERS:", list(df.columns))
        print("DATE CANDIDATES:", date_candidates)
        print("AMOUNT CANDIDATES:", amount_candidates)

        # 7. Initialise the results.
        anomaly_records = []
        output_columns = []
        analysis_message = None

        # 8. Run the algorithm when suitable columns exist.
        if not date_candidates:
            analysis_message = (
                "No date or time column was found. "
                "The anomaly algorithm was not run."
            )

        elif not amount_candidates:
            analysis_message = (
                "No suitable amount column was found. "
                "The anomaly algorithm was not run."
            )

        elif row_count < 3:
            analysis_message = (
                "At least 3 transaction rows are required "
                "for this analysis."
            )

        else:
            date_column = date_candidates[0]
            amount_column = amount_candidates[0]

            print("Running detect_anomalies()")
            print("Date column:", date_column)
            print("Amount column:", amount_column)

            # This function is defined in voucher_anomaly.py.
            anomaly_output = detect_anomalies(
                df,
                date_column=date_column,
                amount_column=amount_column,
                scales=6,
                epsilon=1.0,
                base_scale=10.5,
                omega0=6.0,
                beta=1.0,
                alpha=1.0,
                beta_weight=1.0,
                gamma=1.0,
                diffusion_time=1.0,
            )
            # 9. Convert algorithm output into template-friendly data.
            if isinstance(anomaly_output, pd.DataFrame):
                output_df = anomaly_output.copy()

            elif isinstance(anomaly_output, tuple) and len(anomaly_output) > 0:
                # Support functions that return (DataFrame, diagnostics).
                if isinstance(anomaly_output[0], pd.DataFrame):
                    output_df = anomaly_output[0].copy()
                else:
                    raise ValueError(
                        "The algorithm did not return a DataFrame."
                    )

            else:
                raise ValueError(
                    "Unexpected output from detect_anomalies()."
                )

            # Replace missing/infinite values before rendering.
            output_df = output_df.replace(
                [float("inf"), float("-inf")],
                float("nan"),
            )

            output_df = output_df.astype(object).where(
                pd.notna(output_df),
                None,
            )

            # IMPORTANT: Define output_columns before using it in context.
            output_columns = list(output_df.columns)

            # Each row is a dictionary, so analysis.html can use record.values.
            anomaly_records = output_df.to_dict(orient="records")

            print("ANOMALY DETECTION SUCCESSFUL")
            print("Output columns:", output_columns)
            print("Output rows:", len(anomaly_records))

            analysis_message = (
                "Analysis completed. Review the transaction scores below. "
                "A high anomaly score indicates a transaction that deserves "
                "further examination; it does not prove fraud."
            )

        # 10. Prepare the column inspection report.
        column_report = []

        schema = inspection.get("schema", {})
        missing_values = inspection.get("missing_values", {})
        unique_counts = inspection.get("unique_counts", {})

        for col in inspection.get("columns", []):
            details = schema.get(col, {})

            column_report.append({
                "name": col,
                "type": details.get("type", "Unknown"),
                "confidence": details.get("confidence", ""),
                "missing": missing_values.get(col, 0),
                "unique": unique_counts.get(col, 0),
            })

        # 11. Send the data to analysis.html.
        context = {
            "result": inspection,
            "row_count": row_count,
            "column_count": column_count,
            "columns": inspection.get("columns", []),
            "column_report": column_report,
            "numeric_columns": inspection.get("numeric_columns", []),
            "date_columns": inspection.get("date_columns", []),
            "bool_columns": inspection.get("bool_columns", []),
            "id_columns": inspection.get("id_columns", []),
            "categorical_columns": inspection.get(
                "categorical_columns", []
            ),
            "text_columns": inspection.get("text_columns", []),
            "missing_values": missing_values,
            "missing_percent": inspection.get("missing_percent", {}),
            "unique_counts": unique_counts,
            "numeric_stats": inspection.get("numeric_stats", {}),
            "duplicate_rows": inspection.get("duplicate_rows", 0),
            "quality_score": inspection.get("quality_score", 0),
            "issues": inspection.get("issues", []),
            "preview": inspection.get("preview", []),
            "encoding_used": inspection.get("encoding_used", ""),
            "anomaly_records": anomaly_records,
            "output_columns": output_columns,
            "analysis_message": analysis_message,
        }

        return render(
            request,
            "detector/analysis.html",
            context,
        )

    except (ValueError, UnicodeDecodeError) as exc:
        print("CSV PROCESSING ERROR:", repr(exc))

        return render(
            request,
            "detector/upload.html",
            {"error": str(exc)},
        )

    except Exception as exc:
        # Keep technical details in the Terminal for debugging.
        print("ANALYSIS ERROR:", repr(exc))

        return render(
            request,
            "detector/upload.html",
            {
                "error": (
                    "Analysis failed. Please check the Terminal "
                    "for the error details."
                )
            },
        )