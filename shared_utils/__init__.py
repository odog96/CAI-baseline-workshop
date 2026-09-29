"""
Shared utilities for MLOps workshop
"""

from .config import (
    DATALAKE_CONFIG,
    MODEL_CONFIG,
    FEATURE_CONFIG,
    API_CONFIG,
)

# data_connection no longer imports pyspark at module scope, so this import is
# safe in a runtime with no Spark add-on. Availability is reported by
# spark_available() and enforced by require_spark() at the point of use, which
# gives an actionable message instead of a bare ModuleNotFoundError on import.
from .data_connection import (
    get_connection_name,
    get_data_connection,
    get_spark_session,
    read_table_with_spark,
    require_spark,
    spark_available,
    spark_to_pandas,
)

from .runtime import (
    current_kernel,
    list_available_runtimes,
    resolve_runtime,
)

from .naming import (
    external_warehouse_dir,
    raw_username,
    safe_username,
    workshop_database,
    workshop_database_location,
    workshop_table,
    workshop_table_path,
)

from .model_endpoint import get_model_endpoint_config

_SPARK_AVAILABLE = spark_available()

__all__ = [
    # Config
    'DATALAKE_CONFIG',
    'MODEL_CONFIG',
    'FEATURE_CONFIG',
    'API_CONFIG',
    # Data connections
    'get_connection_name',
    'get_data_connection',
    'get_spark_session',
    'read_table_with_spark',
    'require_spark',
    'spark_available',
    'spark_to_pandas',
    # Runtime discovery
    'current_kernel',
    'list_available_runtimes',
    'resolve_runtime',
    # Per-participant naming
    'external_warehouse_dir',
    'raw_username',
    'safe_username',
    'workshop_database',
    'workshop_database_location',
    'workshop_table',
    'workshop_table_path',
    # Model endpoint credentials
    'get_model_endpoint_config',
]
