"""
Reset Workshop Environment
===========================

This script cleans up the workshop environment by:
1. Deleting the downloaded local data
2. Dropping this participant's workshop database and its tables

Use this to start fresh or clean up after the workshop.

Scope: this only ever touches YOUR OWN database -- the name is derived by
shared_utils.naming from your username, the same way 01_ingest.py derived it
when it created the database. In a shared lab that matters: resetting cannot
take out another participant's data, or a pre-existing site database.
"""

import os
import sys
import shutil

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_utils import (
    get_connection_name,
    require_spark,
    safe_username,
    workshop_database,
)

# Anchored to this file's location, not the working directory -- 01_ingest.py
# writes under module1/, and a cwd-relative "data" would miss it entirely when
# reset is run from the project root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_DATA_DIRS = [
    os.path.join(PROJECT_ROOT, "module1", "data", "bank-additional"),
    os.path.join(PROJECT_ROOT, "module1", "inference_data"),
]
LOCAL_DATA_FILES = [
    os.path.join(PROJECT_ROOT, "module1", "data", "bank-additional.zip"),
]


def delete_local_data():
    """
    Delete the downloaded dataset and derived inference data.

    Deletes the *contents* placed there by 01_ingest.py rather than the
    module1/data directory itself, so the .gitkeep that keeps the directory in
    git survives.
    """
    for path in LOCAL_DATA_DIRS:
        if os.path.isdir(path):
            print(f"Deleting {path}")
            shutil.rmtree(path)
            print(f"✓ Deleted {path}")
        else:
            print(f"✓ Nothing to delete at {path}")

    for path in LOCAL_DATA_FILES:
        if os.path.isfile(path):
            print(f"Deleting {path}")
            os.remove(path)
            print(f"✓ Deleted {path}")
        else:
            print(f"✓ Nothing to delete at {path}")


def drop_database_and_tables():
    """
    Drop this participant's workshop database and all tables in it.
    """
    import cml.data_v1 as cmldata

    try:
        require_spark()
    except Exception as error:
        print(f"⊘ SKIPPED dropping the database: {error}")
        return

    # Same derivation 01_ingest.py used to CREATE it. Previously this was the
    # literal "DEFAULT_ML_WORKSHOP", which matched nothing ingest ever created,
    # so the reset silently did nothing and a re-run of ingest hit a table that
    # was supposedly gone.
    full_db = workshop_database()

    try:
        connection_name = get_connection_name()
        print(f"\nConnecting to data lake: {connection_name}")
        conn = cmldata.get_connection(connection_name)
        spark = conn.get_spark_session()
    except Exception as error:
        print(f"⊘ SKIPPED dropping the database ({type(error).__name__}: {error})")
        print("  Check CONNECTION_NAME in Project Settings > Advanced against the")
        print("  names in Project Settings > Data Connections.")
        return

    try:
        print(f"\nChecking for database: {full_db}")

        databases = spark.sql("SHOW DATABASES").collect()
        db_names = [row.namespace.lower() for row in databases]

        if full_db.lower() in db_names:
            tables = spark.sql(f"SHOW TABLES IN `{full_db}`").collect()
            if tables:
                print(f"Tables in {full_db}:")
                for table in tables:
                    print(f"  - {table.tableName}")

            # CASCADE drops the tables with it.
            print(f"\nDropping database: {full_db}")
            spark.sql(f"DROP DATABASE IF EXISTS `{full_db}` CASCADE")
            print(f"✓ Dropped database: {full_db}")
        else:
            print(f"✓ Database doesn't exist: {full_db}")
    except Exception as error:
        print(f"✗ Error dropping {full_db} ({type(error).__name__}: {error})")
    finally:
        spark.stop()


def main():
    """
    Main reset function
    """
    print("=" * 60)
    print("Workshop Environment Reset")
    print("=" * 60)

    print(f"\nUser: {safe_username()}")

    print("\n⚠️  WARNING: This will delete:")
    for path in LOCAL_DATA_DIRS + LOCAL_DATA_FILES:
        print(f"  - {path}")
    print(f"  - Database: {workshop_database()} (and every table in it)")

    # Run as a CML job there is no stdin, and input() would block forever with
    # the job appearing to hang. Require an explicit opt-in instead.
    if not sys.stdin.isatty():
        if os.environ.get("RESET_CONFIRM", "").strip().lower() != "yes":
            print("\nNo terminal to confirm on (running as a job?).")
            print("Set RESET_CONFIRM=yes to proceed non-interactively.")
            print("\nReset cancelled.")
            return
        print("\nRESET_CONFIRM=yes -- proceeding without a prompt.")
    else:
        response = input("\nContinue? (yes/no): ").strip().lower()
        if response != "yes":
            print("\nReset cancelled.")
            return

    # Step 1: Delete local data
    print("\n" + "-" * 60)
    print("Step 1: Delete Local Data")
    print("-" * 60)
    delete_local_data()

    # Step 2: Drop database and tables
    print("\n" + "-" * 60)
    print("Step 2: Drop Database and Tables")
    print("-" * 60)
    drop_database_and_tables()

    # Summary
    print("\n" + "=" * 60)
    print("Reset Complete!")
    print("=" * 60)
    print("\nYou can now run the workshop from scratch:")
    print("  python module1/01_ingest.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
