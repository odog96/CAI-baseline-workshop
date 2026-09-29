"""
Shared configuration for the MLOps workshop
"""

# Data lake configuration
# Per-participant database and table names are derived in shared_utils/naming.py
# so that every script agrees; these are the base names it composes from.
DATALAKE_CONFIG = {
    "database_name": "ml_workshop",
    "table_name": "bank_marketing",
}

# Model configuration
MODEL_CONFIG = {
    "target_column": "y",
    "test_size": 0.2,
    "random_state": 42,
}

# Feature engineering configuration
FEATURE_CONFIG = {
    # Features to use in model training
    "numeric_features": [
        "age", "balance", "day", "duration", 
        "campaign", "pdays", "previous"
    ],
    "categorical_features": [
        "job", "marital", "education", "default", 
        "housing", "loan", "contact", "month", "poutcome"
    ],
}

# API deployment configuration
API_CONFIG = {
    # Underscores, not hyphens. This is the name module1/04_deploy.py actually
    # deploys under and the name module2/02.2 and module3/5 look up, so a
    # hyphenated value here silently failed to match the deployed model.
    "model_name": "banking_campaign_predictor",
    "description": "Predicts if a customer will subscribe to a term deposit",
    "cpu": 1,
    "memory": 2,  # GB
}

# Model endpoint configuration for inference
#
# Intentionally NOT stored here. A model access key authorises calls to the
# model's REST endpoint, so committing one to git publishes a working credential
# -- and because git keeps history, deleting it later does not un-publish it.
# An access key that previously lived on this line has been removed for exactly
# that reason.
#
# Use shared_utils.get_model_endpoint_config() instead. It resolves the endpoint
# and key at runtime from (in order) environment variables, a git-ignored local
# file, or the CML API -- the API path needs no configuration at all, because
# each participant's own deployed model already holds its key.
#
#     from shared_utils import get_model_endpoint_config
#     config = get_model_endpoint_config()
#     endpoint, key = config["model_endpoint"], config["access_key"]
