# PySpark prediction analytics

`daily_prediction_statistics.py` is the Day 4 offline analytics job. It reads
the `predictions` table from PostgreSQL through Spark JDBC and groups records
by prediction date, model name, model alias, and derived defect status.

Successful records become `defect` or `no_defect` using the probability
threshold. Failed inference records remain visible as `error`. The report
contains category counts, average probability, average latency, total and
successful prediction counts, defect/error counts, and defect/error rates.

Install the optional analytics dependency:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[analytics]"
```

Run it locally with the PostgreSQL JDBC driver supplied to Spark:

```powershell
$env:FACTORYVISION_DATABASE_URL = "postgresql+psycopg://factoryvision:factoryvision@localhost:5432/factoryvision"
.venv\Scripts\spark-submit.cmd `
  --packages org.postgresql:postgresql:42.7.4 `
  spark\daily_prediction_statistics.py
```

The default output is Parquet under
`artifacts/spark/daily_prediction_statistics`. Use `--format csv` or
`--format json` when a human-readable interchange format is more convenient.
The JDBC driver must be available on Spark's classpath because Spark's JDBC
reader does not include database-specific drivers automatically.

On Windows, Spark 3.5.6 may also require a Hadoop `winutils.exe` installation
with `HADOOP_HOME` set when staging JDBC jars or writing output files. If that
is not configured, run the job in a Linux/Docker Spark environment; the
DataFrame transformation tests do not require `winutils.exe`.
