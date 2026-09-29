"""
Data lake connection utilities for Cloudera AI.

Everything here goes through the CML data connection rather than building a
SparkSession by hand. That is the whole point of the connection: it carries the
site's Hive Metastore URI, Kerberos principal/keytab, Ranger policies and
warehouse location. A hand-rolled SparkSession.builder gets none of that, so it
comes up with no metastore and every spark.table() call fails -- which is why
the previous version of get_spark_session() could never have worked on prem.
"""

import importlib.util
import os

import cml.data_v1 as cmldata


def spark_available():
    """True if pyspark is importable.

    pyspark is NOT in the Standard ML Runtime -- it arrives via a Spark runtime
    add-on selected when the session or job is created. This is the single most
    common reason Module 1 fails for a new attendee.
    """
    return importlib.util.find_spec("pyspark") is not None


def require_spark():
    """Raise a legible error instead of a bare ModuleNotFoundError."""
    if not spark_available():
        raise RuntimeError(
            "pyspark is not available in this runtime.\n"
            "Spark is not part of the Standard ML Runtime -- it comes from a "
            "Spark runtime add-on.\n"
            "Fix: stop this session, start a new one, and tick a Spark add-on "
            "(e.g. 'Spark 3.5.4 - CDP 7.3.2.0') in the launch dialog. For jobs, "
            "set the same add-on on the job definition."
        )


def get_connection_name():
    """
    Resolve the Spark data connection name.

    Order of preference:
      1. The CONNECTION_NAME environment variable (Project Settings > Advanced).
      2. Auto-discovery, when the project has exactly one Spark connection.

    Raises a message naming the available connections rather than failing deep
    inside Spark, because the old hardcoded defaults ('go01-aw-dl', 'se-aws-edl',
    'go01-demo') were all public-cloud names that cannot resolve on prem.
    """
    name = os.environ.get("CONNECTION_NAME", "").strip()
    if name:
        return name

    try:
        connections = cmldata.get_project_dataconnections()
    except Exception as error:
        raise RuntimeError(
            "CONNECTION_NAME is not set and the project's data connections "
            f"could not be listed ({error}).\n"
            "Fix: set CONNECTION_NAME in Project Settings > Advanced to the name "
            "shown in Project Settings > Data Connections."
        ) from error

    def field(conn, *keys):
        for key in keys:
            value = conn.get(key) if isinstance(conn, dict) else getattr(conn, key, None)
            if value:
                return str(value)
        return ""

    named = [(field(c, "name", "displayName"), field(c, "type", "connectionType"))
             for c in connections]
    spark = [n for n, t in named if "spark" in t.lower()]

    if len(spark) == 1:
        print(f"CONNECTION_NAME not set; using the only Spark connection: {spark[0]!r}")
        return spark[0]

    listing = "\n".join(f"  {n!r} (type: {t or 'unknown'})" for n, t in named) or "  (none)"
    raise RuntimeError(
        f"Could not pick a Spark data connection automatically "
        f"({len(spark)} Spark connections found).\n"
        "Fix: set CONNECTION_NAME in Project Settings > Advanced to one of:\n"
        f"{listing}"
    )


def get_data_connection(connection_name=None):
    """Get the CML data connection object."""
    return cmldata.get_connection(connection_name or get_connection_name())


def get_spark_session(connection_name=None):
    """
    Return a SparkSession wired to this site's metastore via the data connection.
    """
    require_spark()
    conn = get_data_connection(connection_name)
    return conn.get_spark_session()


def read_table_with_spark(database_name, table_name, connection_name=None):
    """
    Read a table using Spark.

    Args:
        database_name: Name of the database
        table_name: Name of the table

    Returns:
        Spark DataFrame
    """
    spark = get_spark_session(connection_name)
    full_table_name = f"{database_name}.{table_name}"
    return spark.table(full_table_name)


def spark_to_pandas(spark_df):
    """
    Convert Spark DataFrame to Pandas with proper handling

    Args:
        spark_df: Spark DataFrame

    Returns:
        Pandas DataFrame
    """
    return spark_df.toPandas()
