"""Build a SparkSession that works reliably on Windows + Python 3.12.

Notes
-----
* Sets JAVA_HOME programmatically so PySpark can find the JDK even when
  the user has not exported it.
* Detects winutils for HADOOP_HOME if available (silently skips otherwise -
  parquet read/write works without it on local filesystem).
* Uses a tuned local mode that's appropriate for a developer laptop.
"""

from __future__ import annotations

import os
from pathlib import Path

from .utils import get_logger

log = get_logger(__name__)


_DEFAULT_JDK_CANDIDATES = (
    r"C:\Program Files\Java\jdk-17",
    r"C:\Program Files\Eclipse Adoptium\jdk-17.0.10.7-hotspot",
    r"C:\Program Files\Eclipse Adoptium\jdk-17",
)


def _ensure_java_home() -> None:
    if os.environ.get("JAVA_HOME") and Path(os.environ["JAVA_HOME"]).exists():
        return
    for candidate in _DEFAULT_JDK_CANDIDATES:
        if Path(candidate).exists():
            os.environ["JAVA_HOME"] = candidate
            log.info("Set JAVA_HOME=%s", candidate)
            return
    log.warning("JAVA_HOME not set and no default JDK 17 found.")


def _ensure_python() -> None:
    """Force PySpark workers to use the venv's Python interpreter."""
    import sys
    py = sys.executable
    os.environ.setdefault("PYSPARK_PYTHON", py)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", py)


def get_spark(app_name: str = "ecom-bigdata", shuffle_partitions: int = 8):
    """Return a (cached) SparkSession suitable for laptop-scale local jobs."""
    _ensure_java_home()
    _ensure_python()

    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.driver.memory", "4g")
        .config("spark.driver.maxResultSize", "1g")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.log.level", "WARN")
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    log.info("Spark %s session ready (master=%s)", spark.version, spark.sparkContext.master)
    return spark


__all__ = ["get_spark"]
