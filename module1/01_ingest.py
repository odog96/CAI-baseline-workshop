"""
Module 1 - Step 1: Data Ingestion
===================================

This script downloads the UCI Bank Marketing dataset and writes it to the data lake.

Key Learning Points:
- Direct data ingestion without complex transformations
- Using Spark for write operations (leveraging Iceberg format benefits)
- Setting up the foundation for downstream ML workflows

Dataset: UCI Bank Marketing Dataset
https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
"""

import os
import sys
import zipfile
import urllib.request
import pandas as pd
import numpy as np
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_utils import (
    DATALAKE_CONFIG,
    get_data_connection,
    spark_available,
)

# All paths are anchored to this script's directory, never to the working
# directory. A CML job runs with cwd=/home/cdsw while an interactive session in
# the file tree may run with cwd=/home/cdsw/module1 -- so cwd-relative paths put
# the data in two different places depending on how the script was started, and
# the training helpers (which use absolute paths under module1/) then cannot
# find it.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
INFER_DIR = os.path.join(SCRIPT_DIR, "inference_data")
ZIP_PATH = os.path.join(DATA_DIR, "bank-additional.zip")
CSV_PATH = os.path.join(DATA_DIR, "bank-additional", "bank-additional-full.csv")


def download_data(url, local_zip_path, extract_dir):
    """Download and extract the dataset"""
    download_start = time.time()

    print("Downloading dataset...")

    urllib.request.urlretrieve(url, local_zip_path)
    print(f"Downloaded to {local_zip_path}")

    extract_start = time.time()
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    extract_elapsed = time.time() - extract_start

    print(f"Extracted to {extract_dir} ({extract_elapsed:.2f} seconds)")

    download_elapsed = time.time() - download_start
    print(f"Total download time: {download_elapsed:.2f} seconds")



def load_and_inspect_data(csv_path):
    """
    Load CSV into Pandas and perform basic inspection

    Args:
        csv_path: Path to CSV file

    Returns:
        Pandas DataFrame
    """
    load_start = time.time()

    print("\nLoading data with Pandas...")
    print("\n printing csv_path", csv_path)

    read_start = time.time()
    df = pd.read_csv(csv_path, delimiter=';')
    read_elapsed = time.time() - read_start

    print(f"Data loaded in {read_elapsed:.2f} seconds")
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())

    print(f"\nData types:")
    print(df.dtypes)

    print(f"\nTarget variable distribution:")
    print(df['y'].value_counts())

    load_elapsed = time.time() - load_start
    print(f"\nTotal load and inspect time: {load_elapsed:.2f} seconds")

    return df


def write_to_datalake(df, database_name, table_name, username):
    """
    Write Pandas DataFrame to the data lake using Spark.

    This step is a demonstration of the data lake round-trip, and it is
    deliberately NON-FATAL. The CSV is already on the project filesystem by the
    time this runs, and the only downstream consumer of the table is the EDA
    notebook -- every other step in modules 1-4 reads the CSV. So when Spark,
    the data connection, or the metastore is unavailable, skipping costs one
    demonstration rather than the whole workshop.

    Returns:
        The fully-qualified table name on success, or None if skipped.
    """
    write_start = time.time()

    print(f"\nWriting to data lake with user: {username}")

    if not spark_available():
        print("⊘ SKIPPED: pyspark is not available in this runtime.")
        print("  Spark comes from a runtime add-on, not the Standard runtime.")
        print("  To run this step, start a session with a Spark add-on enabled")
        print("  (e.g. 'Spark 3.5.4 - CDP 7.3.2.0').")
        print("  Everything downstream reads the CSV, so the workshop continues.")
        return None

    try:
        conn = get_data_connection()
        spark = conn.get_spark_session()
    except Exception as error:
        print(f"⊘ SKIPPED: could not open a Spark session ({type(error).__name__}: {error})")
        print("  Check CONNECTION_NAME in Project Settings > Advanced against the")
        print("  names in Project Settings > Data Connections.")
        print("  Everything downstream reads the CSV, so the workshop continues.")
        return None

    # pandas 2.0 removed DataFrame.iteritems, which older pyspark still calls
    # from createDataFrame. This shim is load-bearing, not dead code.
    if not hasattr(pd.DataFrame, 'iteritems'):
        pd.DataFrame.iteritems = pd.DataFrame.items

    # Create unique database and table names
    full_db = f"{database_name}_{username}".upper()
    full_table = f"{table_name}_{username}".upper()
    full_path = f"{full_db}.{full_table}"

    try:
        conversion_start = time.time()
        spark_df = spark.createDataFrame(df)
        conversion_elapsed = time.time() - conversion_start
        print(f"Spark DataFrame conversion: {conversion_elapsed:.2f} seconds")

        spark.sql(f"CREATE DATABASE IF NOT EXISTS {full_db}")

        # Prefer Iceberg. Fall back to a plain managed table if the Iceberg
        # extensions are not configured on this connection -- the workshop only
        # needs the data to be queryable, and the table format is incidental to
        # every downstream step.
        try:
            iceberg_start = time.time()
            spark_df.writeTo(full_path).using("iceberg").createOrReplace()
            print(f"✓ Data written to {full_path} as ICEBERG "
                  f"({time.time() - iceberg_start:.2f} seconds)")
        except Exception as iceberg_error:
            print(f"! Iceberg write failed ({type(iceberg_error).__name__}: {iceberg_error})")
            print("  Falling back to a standard Hive table...")
            hive_start = time.time()
            spark_df.write.mode("overwrite").saveAsTable(full_path)
            print(f"✓ Data written to {full_path} as a HIVE TABLE "
                  f"({time.time() - hive_start:.2f} seconds)")
    except Exception as error:
        print(f"⊘ SKIPPED: write to {full_path} failed "
              f"({type(error).__name__}: {error})")
        print("  Most likely a Ranger permission on CREATE DATABASE, or a")
        print("  metastore warehouse path the workshop user cannot write to.")
        print("  Everything downstream reads the CSV, so the workshop continues.")
        return None

    write_elapsed = time.time() - write_start
    print(f"Total data lake write time: {write_elapsed:.2f} seconds")
    return full_path


def create_sample_inference_data(df):
    """
    Create sample inference data for demonstration and testing purposes.

    Takes a random sample of the raw data (without target variable) and saves
    it to the inference_data folder for use in the inference job pipeline.

    Args:
        df: Full dataset DataFrame
    """
    inference_start = time.time()

    print("\nCreating sample inference data...")

    # Create inference_data directory if it doesn't exist
    os.makedirs(INFER_DIR, exist_ok=True)

    # Take a random sample (1000 records for demo)
    np.random.seed(42)
    sample_df = df.sample(n=min(1000, len(df)), random_state=42)

    # Remove target variable 'y' for inference (in real scenario, we wouldn't have it)
    inference_df = sample_df.drop(columns=['y'])

    # Save as CSV with semicolon delimiter (matching training data format)
    output_path = os.path.join(INFER_DIR, "raw_inference_data.csv")
    inference_df.to_csv(output_path, sep=";", index=False)

    inference_elapsed = time.time() - inference_start

    print(f"✓ Sample inference data created")
    print(f"  Shape: {inference_df.shape}")
    print(f"  Columns: {len(inference_df.columns)}")
    print(f"  Saved to: {output_path}")
    print(f"  Creation time: {inference_elapsed:.2f} seconds")

def main():
    """
    Main execution function with timing instrumentation
    """
    # Start overall timing
    script_start = time.time()

    print("=" * 60)
    print("Module 1 - Step 1: Data Ingestion")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    USERNAME = os.environ.get('HADOOP_USER_NAME') # use HADOOP_USER_NAME env to grab executing user if available
    if not USERNAME:
        USERNAME = os.environ["PROJECT_OWNER"] # fallback to project owner
    print(f"User: {USERNAME}")

    # Configuration
    DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"

    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)

    # Step 1: Download data
    print("\n" + "-" * 60)
    print("STEP 1: Download Data")
    print("-" * 60)
    # The skip-check tests the same path that step 2 reads. It used to test
    # data/bank-additional-full.csv, which nothing ever creates, so the download
    # repeated on every run and a network failure surfaced as a confusing
    # FileNotFoundError from read_csv instead of a download error.
    if not os.path.exists(CSV_PATH):
        download_data(DATA_URL, ZIP_PATH, DATA_DIR)
    else:
        print(f"Data already exists at {CSV_PATH}")

    # Step 2: Load and inspect
    print("\n" + "-" * 60)
    print("STEP 2: Load and Inspect Data")
    print("-" * 60)
    df = load_and_inspect_data(CSV_PATH)

    # Step 3: Write to data lake
    print("\n" + "-" * 60)
    print("STEP 3: Write to Data Lake")
    print("-" * 60)
    table_path = write_to_datalake(
        df,
        DATALAKE_CONFIG["database_name"],
        DATALAKE_CONFIG["table_name"],
        USERNAME  # Add username parameter
    )

    # Step 4: Create sample inference data
    print("\n" + "-" * 60)
    print("STEP 4: Create Sample Inference Data")
    print("-" * 60)
    create_sample_inference_data(df)

    # Calculate total execution time
    script_elapsed = time.time() - script_start

    print("\n" + "=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {script_elapsed:.2f} seconds ({script_elapsed/60:.2f} minutes)")
    print("=" * 60)
    print("✅ Ingestion complete!")
    print(f"\nLocal data (used by every downstream step):")
    print(f"  • {CSV_PATH}")
    print(f"  • {os.path.join(INFER_DIR, 'raw_inference_data.csv')}")
    if table_path:
        print(f"Data lake table (used by 02_eda_notebook.ipynb):\n  • {table_path}")
    else:
        print("Data lake table: skipped -- see STEP 3 above. This is not fatal;")
        print("  02_eda_notebook.ipynb falls back to reading the CSV.")
    print("\nNext steps:")
    print("  • Training pipeline: 02_eda_feature_engineering.ipynb → 03_train_quick.py")
    print("  • Inference pipeline: 05_inference_data_prep.py → 06_inference_predict.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
