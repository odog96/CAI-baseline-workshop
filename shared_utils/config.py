"""
Shared configuration for the MLOps workshop
"""

# Data lake configuration
DATALAKE_CONFIG = {
    "database_name": "default",
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
    "model_name": "banking-campaign-predictor",
    "description": "Predicts if a customer will subscribe to a term deposit",
    "cpu": 1,
    "memory": 2,  # GB
}

# Model endpoint configuration for inference
# UPDATE THESE VALUES with your deployed model's endpoint and access key
MODEL_ENDPOINT_CONFIG = {
    "model_endpoint": "https://modelservice.ml-56979638-3f1.go01-dem.ylcu-atmi.cloudera.site/model",
    "access_key": "m9p35t2ejlml2sbrbumjgo1xt23x2vr0"
}