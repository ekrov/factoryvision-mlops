"""Compute daily FactoryVision prediction statistics with PySpark.

The job reads the PostgreSQL ``predictions`` table through Spark JDBC,
classifies successful predictions using the configured probability threshold,
and writes a partitioned Spark output directory containing daily statistics.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from factoryvision.storage.database import database_url_from_environment


DEFAULT_OUTPUT_PATH = Path("artifacts/spark/daily_prediction_statistics")
DEFAULT_JDBC_DRIVER = "org.postgresql.Driver"
DEFAULT_DEFECT_THRESHOLD = 0.5
_TABLE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def database_url_to_jdbc(database_url: str) -> tuple[str, dict[str, str]]:
    """Convert the project's SQLAlchemy PostgreSQL URL to Spark JDBC settings."""

    normalized_url = database_url
    for prefix in ("postgresql+psycopg://", "postgresql+psycopg2://"):
        if normalized_url.startswith(prefix):
            normalized_url = "postgresql://" + normalized_url[len(prefix) :]
            break
    parsed = urlsplit(normalized_url)
    if parsed.scheme != "postgresql" or not parsed.hostname:
        raise ValueError(
            "database URL must be a PostgreSQL URL such as "
            "postgresql+psycopg://user:password@host:5432/database"
        )
    database_name = parsed.path.lstrip("/")
    if not database_name:
        raise ValueError("database URL must include a database name")
    port = parsed.port or 5432
    jdbc_url = f"jdbc:postgresql://{parsed.hostname}:{port}/{database_name}"
    properties = {
        "user": unquote(parsed.username or ""),
        "password": unquote(parsed.password or ""),
    }
    return jdbc_url, properties


def read_prediction_logs(
    spark: SparkSession,
    database_url: str,
    table_name: str = "predictions",
    jdbc_driver: str = DEFAULT_JDBC_DRIVER,
) -> DataFrame:
    """Read prediction events from PostgreSQL using Spark's JDBC source."""

    if not _TABLE_NAME_PATTERN.fullmatch(table_name):
        raise ValueError(f"unsupported table name: {table_name!r}")
    jdbc_url, properties = database_url_to_jdbc(database_url)
    properties["driver"] = jdbc_driver
    return spark.read.jdbc(
        url=jdbc_url,
        table=table_name,
        properties=properties,
    )


def derive_daily_statistics(
    prediction_logs: DataFrame,
    defect_threshold: float = DEFAULT_DEFECT_THRESHOLD,
) -> DataFrame:
    """Aggregate daily statistics by model version and predicted defect status.

    ``defect_status`` is derived from the model probability for successful
    predictions. Failed inference events are retained as ``error`` rows, which
    allows the report to show both model defect rate and operational error rate.
    """

    if not 0.0 <= defect_threshold <= 1.0:
        raise ValueError("defect_threshold must be between 0 and 1")
    required_columns = {
        "created_at",
        "model_name",
        "model_alias",
        "defect_probability",
        "latency_ms",
        "status",
    }
    missing_columns = required_columns.difference(prediction_logs.columns)
    if missing_columns:
        raise ValueError(f"prediction logs are missing columns: {sorted(missing_columns)}")

    classified = (
        prediction_logs.filter(F.col("created_at").isNotNull())
        .withColumn("prediction_date", F.to_date("created_at"))
        .withColumn("normalized_status", F.lower(F.col("status")))
        .withColumn(
            "defect_status",
            F.when(F.col("normalized_status") != F.lit("success"), F.lit("error"))
            .when(
                F.col("defect_probability") >= F.lit(defect_threshold),
                F.lit("defect"),
            )
            .otherwise(F.lit("no_defect")),
        )
    )

    grouped = (
        classified.groupBy(
            "prediction_date",
            "model_name",
            "model_alias",
            "defect_status",
        )
        .agg(
            F.count(F.lit(1)).alias("prediction_count"),
            F.avg("defect_probability").alias("average_defect_probability"),
            F.avg("latency_ms").alias("average_latency_ms"),
        )
    )
    model_day_window = Window.partitionBy(
        "prediction_date", "model_name", "model_alias"
    )
    return (
        grouped.withColumn(
            "total_predictions",
            F.sum("prediction_count").over(model_day_window),
        )
        .withColumn(
            "successful_predictions",
            F.sum(
                F.when(F.col("defect_status") != "error", F.col("prediction_count"))
                .otherwise(F.lit(0))
            ).over(model_day_window),
        )
        .withColumn(
            "defect_predictions",
            F.sum(
                F.when(F.col("defect_status") == "defect", F.col("prediction_count"))
                .otherwise(F.lit(0))
            ).over(model_day_window),
        )
        .withColumn(
            "error_predictions",
            F.sum(
                F.when(F.col("defect_status") == "error", F.col("prediction_count"))
                .otherwise(F.lit(0))
            ).over(model_day_window),
        )
        .withColumn(
            "defect_rate",
            F.when(
                F.col("successful_predictions") > 0,
                F.col("defect_predictions") / F.col("successful_predictions"),
            ),
        )
        .withColumn(
            "error_rate",
            F.when(
                F.col("total_predictions") > 0,
                F.col("error_predictions") / F.col("total_predictions"),
            ),
        )
        .select(
            "prediction_date",
            "model_name",
            "model_alias",
            "defect_status",
            "prediction_count",
            "average_defect_probability",
            "average_latency_ms",
            "total_predictions",
            "successful_predictions",
            "defect_predictions",
            "error_predictions",
            "defect_rate",
            "error_rate",
        )
        .orderBy("prediction_date", "model_name", "model_alias", "defect_status")
    )


def write_statistics(
    statistics: DataFrame,
    output_path: str | Path,
    output_format: str = "parquet",
) -> None:
    """Write the aggregated DataFrame as a Spark output directory."""

    if output_format not in {"parquet", "json", "csv"}:
        raise ValueError("output_format must be one of: parquet, json, csv")
    writer = statistics.write.mode("overwrite")
    if output_format == "csv":
        writer.option("header", "true").csv(str(output_path))
    else:
        writer.format(output_format).save(str(output_path))


def create_spark_session() -> SparkSession:
    """Create a session using the active Python interpreter on local machines."""

    python_executable = os.environ.get("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_PYTHON", python_executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", python_executable)
    return (
        SparkSession.builder.appName("FactoryVisionDailyPredictionStatistics")
        .config("spark.pyspark.python", python_executable)
        .config("spark.pyspark.driver.python", python_executable)
        .getOrCreate()
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=database_url_from_environment(),
        help="SQLAlchemy PostgreSQL URL; defaults to FACTORYVISION_DATABASE_URL.",
    )
    parser.add_argument(
        "--table",
        default="predictions",
        dest="table_name",
        help="Prediction table to read.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Spark output directory for the daily statistics.",
    )
    parser.add_argument(
        "--format",
        default="parquet",
        choices=("parquet", "json", "csv"),
        dest="output_format",
        help="Output format; Parquet is the default for downstream analytics.",
    )
    parser.add_argument(
        "--defect-threshold",
        type=float,
        default=float(os.getenv("FACTORYVISION_THRESHOLD", DEFAULT_DEFECT_THRESHOLD)),
        help="Probability at or above which a success is classified as defect.",
    )
    parser.add_argument(
        "--jdbc-driver",
        default=DEFAULT_JDBC_DRIVER,
        help="JDBC driver class available on Spark's classpath.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spark = create_spark_session()
    try:
        logs = read_prediction_logs(
            spark,
            database_url=args.database_url,
            table_name=args.table_name,
            jdbc_driver=args.jdbc_driver,
        )
        statistics = derive_daily_statistics(
            logs,
            defect_threshold=args.defect_threshold,
        )
        statistics.show(50, truncate=False)
        write_statistics(statistics, args.output, args.output_format)
        print(f"Wrote daily prediction statistics to {args.output}")
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
