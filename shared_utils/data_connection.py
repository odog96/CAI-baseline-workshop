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


def _field(conn, *keys):
    """Read the first present key, tolerating dict or object shaped entries."""
    for key in keys:
        value = conn.get(key) if isinstance(conn, dict) else getattr(conn, key, None)
        if value:
            return str(value)
    return ""


def list_connection_names():
    """
    Return [(name, type)] for this project's data connections, or [] if they
    cannot be listed.

    Only the 'name' and 'type' fields are read. These entries also carry a
    'connectionInfo' field holding live credentials, which is deliberately never
    touched, logged or returned.
    """
    try:
        connections = cmldata.get_project_dataconnections()
    except Exception:
        return []
    return [(_field(c, "name", "displayName"), _field(c, "type", "connectionType"))
            for c in connections]


def get_connection_name():
    """
    Resolve the Spark data connection name.

    Order of preference:
      1. CONNECTION_NAME (Project Settings > Advanced) -- but only if it actually
         names a connection this project has.
      2. Auto-discovery, when the project has exactly one Spark connection.

    The validation in step 1 is the important part. This project shipped with
    CONNECTION_NAME=go01-aw-dl, an AWS public-cloud Data Lake that does not exist
    on prem. Returning an env value unchecked meant that dead string beat working
    auto-discovery, and the failure surfaced deep inside Spark. Validating it
    means a stale value from an older install, or a typo, self-heals with one
    clear line instead.
    """
    name = os.environ.get("CONNECTION_NAME", "").strip()
    connections = list_connection_names()
    known = [n for n, _ in connections]

    if name:
        if not connections:
            # Cannot verify -- trust the operator rather than block on an API
            # that may simply be unreachable from a job.
            return name
        if name in known:
            ctype = next(t for n, t in connections if n == name)
            if "spark" not in ctype.lower():
                print(f"! CONNECTION_NAME={name!r} is a {ctype} connection, not SPARK.")
                print("  get_spark_session() will likely fail. Set it to a Spark")
                print("  connection in Project Settings > Advanced.")
            return name
        print(f"! CONNECTION_NAME={name!r} matches no data connection in this project.")
        print("  Ignoring it and auto-discovering instead.")
        print("  To silence this, update CONNECTION_NAME in Project Settings > Advanced.")

    if not connections:
        raise RuntimeError(
            "CONNECTION_NAME is not set and the project's data connections could "
            "not be listed.\n"
            "Fix: set CONNECTION_NAME in Project Settings > Advanced to the name "
            "shown in Project Settings > Data Connections."
        )

    spark = [n for n, t in connections if "spark" in t.lower()]

    if len(spark) == 1:
        print(f"Using the project's only Spark connection: {spark[0]!r}")
        return spark[0]

    listing = "\n".join(f"  {n!r} (type: {t or 'unknown'})" for n, t in connections)
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
