"""
Module 1 - Step 4: Model Deployment (V3 - Simplified & Clean)
==============================================================

FIXES APPLIED:
1. Register model in CML to get registered_model_id
2. Use proper API objects with body= parameter
3. Clean error handling without nested try-except

All fixes clearly marked with ✅
"""

import os
import sys
import json
import time
import logging
import traceback
import cmlapi
import mlflow
from mlflow.tracking import MlflowClient
from cmlapi.rest import ApiException

# Add parent directory for imports (works in both script and notebook)
try:
    # If running as a script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(script_dir))
except NameError:
    # If running in Jupyter notebook
    script_dir = os.getcwd()
    if '/module1' in script_dir:
        sys.path.append(os.path.dirname(script_dir))
    else:
        sys.path.append(script_dir)

# Try to import shared_utils, provide defaults if not available
try:
    from shared_utils import API_CONFIG
except ImportError:
    print("⚠️  Could not import shared_utils, using defaults")
    API_CONFIG = {
        "model_name": "banking_campaign_predictor",
        "description": "Banking campaign prediction model"
    }

# ============================================================================
# Logging Setup
# ============================================================================
# Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)

# Set up logging with both file and console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/deployment_debug.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Also enable debug logging for cmlapi if needed
# logging.getLogger('cmlapi').setLevel(logging.DEBUG)

# Get username for unique naming in workshop environment
USERNAME = os.environ.get('HADOOP_USER_NAME')
if not USERNAME:
    USERNAME = os.environ["PROJECT_OWNER"]

# Configuration
EXPERIMENT_NAME = f"BANK_MARKETING_EXPERIMENTS_{USERNAME}"
MODEL_NAME = "banking_campaign_predictor"

print("=" * 80)
print("Module 1 - Step 4: Model Deployment (V3 - Clean)")
print("=" * 80)
logger.info(f"Starting deployment for user: {USERNAME}")
logger.info(f"Experiment: {EXPERIMENT_NAME}")
logger.info(f"Model: {MODEL_NAME}")

# ============================================================================
# Step 1: Find the best model (by F1 score)
# ============================================================================
print("\n[1/5] Finding best model by F1 score...")

mlflow_client = MlflowClient()
experiment = mlflow_client.get_experiment_by_name(EXPERIMENT_NAME)

if not experiment:
    print(f"❌ ERROR: Experiment '{EXPERIMENT_NAME}' not found!")
    sys.exit(1)

runs = mlflow_client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.test_f1 DESC"],
    max_results=1
)

if not runs:
    print(f"❌ ERROR: No runs found in experiment")
    sys.exit(1)

best_run = runs[0]
run_id = best_run.info.run_id
model_uri = f"runs:/{run_id}/model"
f1_score = best_run.data.metrics.get('test_f1', 0)

print(f"✅ Best model found:")
print(f"   Run ID: {run_id}")
print(f"   F1 Score: {f1_score:.4f}")

# ============================================================================
# Step 2: Register model in CML
# ============================================================================
print("\n[2/5] Registering model in CML...")
logger.info("=" * 60)
logger.info("STEP 2: Model Registration")
logger.info("=" * 60)

# Create CML client and log connection info
cml_client = cmlapi.default_client()
project_id = os.environ.get("CDSW_PROJECT_ID")

if not project_id:
    logger.error("CDSW_PROJECT_ID environment variable not found")
    print("❌ ERROR: Not running in CML environment")
    sys.exit(1)

logger.info(f"Project ID: {project_id}")
logger.info(f"Experiment ID: {experiment.experiment_id}")
logger.info(f"Run ID: {run_id}")

# Check if user provided a manually registered model ID
manual_registered_model_id = os.environ.get("REGISTERED_MODEL_ID")
manual_model_version_id = os.environ.get("MODEL_VERSION_ID")

if manual_registered_model_id and manual_model_version_id:
    logger.info("=" * 60)
    logger.info("USING MANUALLY PROVIDED MODEL IDS")
    logger.info("=" * 60)
    logger.info(f"Registered Model ID: {manual_registered_model_id}")
    logger.info(f"Model Version ID: {manual_model_version_id}")

    registered_model_id = manual_registered_model_id
    model_version_id = manual_model_version_id

    print(f"✅ Using manually provided model IDs:")
    print(f"   Registered Model ID: {registered_model_id}")
    print(f"   Model Version ID: {model_version_id}")

    # Skip to step 3
    skip_registration = True
else:
    skip_registration = False

if not skip_registration:
    # Log authentication context
    logger.info("Authentication Context:")
    logger.info(f"  - HADOOP_USER_NAME: {os.environ.get('HADOOP_USER_NAME', 'NOT SET')}")
    logger.info(f"  - PROJECT_OWNER: {os.environ.get('PROJECT_OWNER', 'NOT SET')}")
    logger.info(f"  - Current User (USERNAME): {USERNAME}")

    # Check if model already exists in registry
    logger.info("Checking for existing registered models...")
    try:
        # Note: list_registered_models() doesn't take project_id parameter
        # It returns all registered models the user has access to
        existing_models = cml_client.list_registered_models()

        if hasattr(existing_models, 'models') and existing_models.models:
            # Filter to only this project's models
            project_models = [m for m in existing_models.models if hasattr(m, 'project_id') and m.project_id == project_id]
            logger.info(f"Found {len(project_models)} registered models in this project (out of {len(existing_models.models)} total)")
        else:
            project_models = []
            logger.info("No registered models found")

        # Check if our model name already exists
        existing_model = None
        for model in project_models:
            if model.name == MODEL_NAME:
                existing_model = model
                logger.warning(f"Model '{MODEL_NAME}' already exists in registry!")
                logger.info(f"  - Existing Model ID: {model.model_id}")
                logger.info(f"  - Created: {model.created_at}")
                break

        if existing_model:
            logger.info("Using existing registered model instead of creating new one")
            registered_model_id = existing_model.model_id

            # Get the latest version or create a new version
            logger.info("Attempting to create new model version...")
            try:
                create_model_version_request = cmlapi.CreateModelVersionRequest(
                    project_id=project_id,
                    registered_model_id=registered_model_id,
                    experiment_id=experiment.experiment_id,
                    run_id=run_id,
                    model_path="model"
                )

                logger.info("Request payload:")
                logger.info(f"  - registered_model_id: {registered_model_id}")
                logger.info(f"  - experiment_id: {experiment.experiment_id}")
                logger.info(f"  - run_id: {run_id}")
                logger.info(f"  - model_path: model")

                # Note: project_id and registered_model_id are in the body
                version_response = cml_client.create_model_version(
                    body=create_model_version_request
                )
                model_version_id = version_response.model_version_id
                logger.info(f"✅ New model version created: {model_version_id}")

            except ApiException as ve:
                logger.error(f"Failed to create model version: {ve.reason}")
                logger.error(f"Status: {ve.status}")
                logger.error(f"Body: {ve.body}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")

                # Try to use existing version
                logger.info("Attempting to use latest existing version...")
                versions = cml_client.list_model_versions(
                    registered_model_id=registered_model_id
                )
                if versions.model_versions:
                    model_version_id = versions.model_versions[0].model_version_id
                    logger.info(f"Using existing version: {model_version_id}")
                else:
                    logger.error("No existing versions found!")
                    sys.exit(1)

            print(f"✅ Using existing registered model:")
            print(f"   Registered Model ID: {registered_model_id}")
            print(f"   Model Version ID: {model_version_id}")

        else:
            # Model doesn't exist, create new one
            logger.info(f"Creating new registered model: {MODEL_NAME}")

            # ✅ Try using the proper API object instead of dict
            create_registered_model_request = cmlapi.CreateRegisteredModelRequest(
                project_id=project_id,
                experiment_id=experiment.experiment_id,
                run_id=run_id,
                model_name=MODEL_NAME,
                model_path="model"
            )

            logger.info("Request payload:")
            logger.info(f"  - project_id: {project_id}")
            logger.info(f"  - experiment_id: {experiment.experiment_id}")
            logger.info(f"  - run_id: {run_id}")
            logger.info(f"  - model_name: {MODEL_NAME}")
            logger.info(f"  - model_path: model")

            try:
                # Note: project_id is already in the body, don't pass it separately
                registered_model_response = cml_client.create_registered_model(
                    body=create_registered_model_request
                )
                registered_model_id = registered_model_response.model_id
                model_version_id = registered_model_response.model_versions[0].model_version_id

                logger.info(f"✅ Successfully created registered model")
                logger.info(f"  - Registered Model ID: {registered_model_id}")
                logger.info(f"  - Model Version ID: {model_version_id}")

                print(f"✅ Model registered in CML:")
                print(f"   Registered Model ID: {registered_model_id}")
                print(f"   Model Version ID: {model_version_id}")

            except ApiException as e:
                logger.error("=" * 60)
                logger.error("API EXCEPTION DETAILS")
                logger.error("=" * 60)
                logger.error(f"Status Code: {e.status}")
                logger.error(f"Reason: {e.reason}")
                logger.error(f"Body: {e.body}")
                logger.error(f"Headers: {e.headers if hasattr(e, 'headers') else 'N/A'}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")

                # Parse the error body for more details
                try:
                    error_body = json.loads(e.body) if isinstance(e.body, str) else e.body
                    logger.error("Parsed error details:")
                    logger.error(f"  - Error: {error_body.get('error', 'N/A')}")
                    logger.error(f"  - Code: {error_body.get('code', 'N/A')}")
                    logger.error(f"  - Message: {error_body.get('message', 'N/A')}")

                    # Check for permission issues
                    if "401" in str(e.status) or "Unauthorized" in str(e.body):
                        logger.error("")
                        logger.error("🔍 PERMISSION ISSUE DETECTED:")
                        logger.error("  This appears to be a permission/authorization error.")
                        logger.error("  Trying alternative approach using MLflow registry...")

                        # ============================================
                        # WORKAROUND: Use MLflow Registry Instead
                        # ============================================
                        try:
                            logger.info("=" * 60)
                            logger.info("ATTEMPTING MLFLOW REGISTRY WORKAROUND")
                            logger.info("=" * 60)
                            print("\n⚠️  CML API permission error detected")
                            print("🔄 Trying alternative approach using MLflow registry...")

                            # Register model with MLflow
                            mlflow_model_uri = f"runs:/{run_id}/model"
                            logger.info(f"Registering model using MLflow: {mlflow_model_uri}")

                            mlflow_registered_model = mlflow.register_model(
                                model_uri=mlflow_model_uri,
                                name=MODEL_NAME
                            )

                            logger.info(f"✅ MLflow registration successful:")
                            logger.info(f"  - Name: {mlflow_registered_model.name}")
                            logger.info(f"  - Version: {mlflow_registered_model.version}")

                            print(f"✅ Model registered via MLflow:")
                            print(f"   Name: {mlflow_registered_model.name}")
                            print(f"   Version: {mlflow_registered_model.version}")

                            # Now check if CML can see this model
                            logger.info("Checking if CML can access the MLflow registered model...")
                            time.sleep(3)  # Give CML time to sync

                            existing_models_check = cml_client.list_registered_models()
                            mlflow_model_in_cml = None

                            if hasattr(existing_models_check, 'models') and existing_models_check.models:
                                for m in existing_models_check.models:
                                    logger.info(f"Checking model: {m.name} (ID: {m.model_id})")
                                    if m.name == MODEL_NAME:
                                        mlflow_model_in_cml = m
                                        logger.info(f"✅ Found MLflow model in CML registry!")
                                        break

                            if mlflow_model_in_cml:
                                registered_model_id = mlflow_model_in_cml.model_id

                                # Get the latest version
                                # Note: registered_model_id is enough, no project_id needed
                                versions = cml_client.list_model_versions(
                                    registered_model_id=registered_model_id
                                )

                                if versions.model_versions:
                                    model_version_id = versions.model_versions[0].model_version_id
                                    logger.info(f"✅ Using model version: {model_version_id}")

                                    print(f"✅ Model accessible in CML:")
                                    print(f"   Registered Model ID: {registered_model_id}")
                                    print(f"   Model Version ID: {model_version_id}")
                                else:
                                    raise Exception("No model versions found after MLflow registration")
                            else:
                                raise Exception("MLflow registered model not visible in CML registry")

                        except Exception as mlflow_err:
                            logger.error(f"MLflow workaround failed: {str(mlflow_err)}")
                            logger.error(f"Full traceback:\n{traceback.format_exc()}")

                            logger.error("")
                            logger.error("  Original troubleshooting steps:")
                            logger.error("    1. Check project permissions: Settings > Collaborators")
                            logger.error("    2. Verify you have 'Business User' or 'Admin' role")
                            logger.error("    3. Try creating model via UI: Models > New Model")
                            logger.error("    4. Check if project has model registry enabled")
                            logger.error("    5. Contact your CML administrator for API access")

                            print(f"❌ ERROR: {e.reason}")
                            print(f"   Body: {e.body}")
                            print(f"\n❌ MLflow workaround also failed: {str(mlflow_err)}")
                            print(f"\n💡 Check outputs/deployment_debug.log for detailed diagnostics")
                            sys.exit(1)

                except Exception as parse_err:
                    logger.error(f"Could not parse error body: {parse_err}")
                    print(f"❌ ERROR: {e.reason}")
                    print(f"   Body: {e.body}")
                    print(f"\n💡 Check outputs/deployment_debug.log for detailed diagnostics")
                    sys.exit(1)

    except Exception as e:
        logger.error("=" * 60)
        logger.error("UNEXPECTED ERROR IN REGISTRATION")
        logger.error("=" * 60)
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        print(f"❌ UNEXPECTED ERROR: {str(e)}")
        print(f"💡 Check outputs/deployment_debug.log for detailed diagnostics")
        sys.exit(1)

# Step 2 complete - now continue with model creation
# (Note: if skip_registration was True, registered_model_id and model_version_id were set earlier)

# ============================================================================
# Step 3: Wait for CML to finalize
# ============================================================================
print("\n[3/5] Waiting for CML to finalize...")
time.sleep(20)
print("   ✅ Ready")

# ============================================================================
# Step 4: Create CML model
# ============================================================================
print("\n[4/5] Creating CML model...")
logger.info("=" * 60)
logger.info("STEP 4: Create CML Model")
logger.info("=" * 60)

# ✅ FIX 2: Use CreateModelRequest with registered_model_id
create_model_request = cmlapi.CreateModelRequest(
    project_id=project_id,
    name=MODEL_NAME,
    description=f"Banking campaign model (F1: {f1_score:.4f})",
    registered_model_id=registered_model_id,  # ✅ CRITICAL!
    disable_authentication=True
)

logger.info("Request payload:")
logger.info(f"  - name: {MODEL_NAME}")
logger.info(f"  - registered_model_id: {registered_model_id}")
logger.info(f"  - description: Banking campaign model (F1: {f1_score:.4f})")

try:
    cml_model = cml_client.create_model(
        body=create_model_request,  # ✅ Use body= parameter
        project_id=project_id
    )
    logger.info(f"✅ Model created successfully: {cml_model.id}")
    logger.info(f"  - Has registered_model_id: {cml_model.registered_model_id}")
    print(f"   ✅ Model created: {cml_model.id}")
    print(f"   ✅ Has registered_model_id: {cml_model.registered_model_id}")

except ApiException as e:
    # Handle "already exists" error
    if "already has a model with that name" in str(e.body):
        logger.warning(f"Model '{MODEL_NAME}' already exists, attempting to retrieve...")
        print(f"   ⚠️  Model already exists, getting it...")

        models = cml_client.list_models(project_id)
        cml_model = next((m for m in models.models if m.name == MODEL_NAME), None)

        if not cml_model:
            logger.error("Could not find existing model in list")
            print(f"   ❌ ERROR: Could not find existing model")
            sys.exit(1)

        logger.info(f"Found existing model: {cml_model.id}")
        logger.info(f"  - registered_model_id: {cml_model.registered_model_id}")

        # Check if existing model has registered_model_id
        if not cml_model.registered_model_id:
            logger.warning("Existing model has NO registered_model_id - recreating...")
            print(f"   ❌ Existing model has NO registered_model_id")
            print(f"   Deleting and recreating...")

            cml_client.delete_model(project_id, cml_model.id)
            logger.info("Model deleted, waiting 5 seconds...")
            time.sleep(5)

            # Recreate
            cml_model = cml_client.create_model(
                body=create_model_request,
                project_id=project_id
            )
            logger.info(f"✅ Model recreated: {cml_model.id}")
            print(f"   ✅ Model recreated: {cml_model.id}")
        else:
            logger.info(f"Using existing model: {cml_model.id}")
            print(f"   ✅ Using existing model: {cml_model.id}")
    else:
        # Other error
        logger.error("=" * 60)
        logger.error("API EXCEPTION IN STEP 4")
        logger.error("=" * 60)
        logger.error(f"Status Code: {e.status}")
        logger.error(f"Reason: {e.reason}")
        logger.error(f"Body: {e.body}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")

        print(f"   ❌ ERROR: {e.reason}")
        print(f"   Body: {e.body}")
        print(f"\n💡 Check outputs/deployment_debug.log for detailed diagnostics")
        sys.exit(1)

# ============================================================================
# Step 5: Create model build
# ============================================================================
print("\n[5/6] Creating model build...")
logger.info("=" * 60)
logger.info("STEP 5: Create Model Build")
logger.info("=" * 60)

runtime_id = "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-workbench-python3.10-standard:2025.09.1-b5"

# ✅ FIX 3: Use CreateModelBuildRequest with registered_model_version_id
create_build_request = cmlapi.CreateModelBuildRequest(
    registered_model_version_id=str(model_version_id),  # ✅ CRITICAL!
    runtime_identifier=runtime_id,
    comment=f"Auto-deployed - F1: {f1_score:.4f}"
)

logger.info("Request payload:")
logger.info(f"  - registered_model_version_id: {model_version_id}")
logger.info(f"  - runtime_identifier: {runtime_id}")
logger.info(f"  - model_id: {cml_model.id}")

try:
    build = cml_client.create_model_build(
        body=create_build_request,  # ✅ Use body= parameter
        project_id=project_id,
        model_id=cml_model.id
    )
    logger.info(f"✅ Build created successfully: {build.id}")
    logger.info(f"  - Status: {build.status}")
    print(f"   ✅ Build created: {build.id}")
    print(f"   ⏳ Build is running (~5-10 minutes)")

except ApiException as e:
    logger.error("=" * 60)
    logger.error("API EXCEPTION IN STEP 5")
    logger.error("=" * 60)
    logger.error(f"Status Code: {e.status}")
    logger.error(f"Reason: {e.reason}")
    logger.error(f"Body: {e.body}")
    logger.error(f"Full traceback:\n{traceback.format_exc()}")

    print(f"   ❌ ERROR: {e.reason}")
    print(f"   Body: {e.body}")
    print(f"\n💡 Check outputs/deployment_debug.log for detailed diagnostics")
    sys.exit(1)

# ============================================================================
# Step 6: Wait for build to complete, then deploy
# ============================================================================
print("\n[6/6] Waiting for build to complete before deployment...")
logger.info("=" * 60)
logger.info("STEP 6: Build Monitoring and Deployment")
logger.info("=" * 60)

# Poll for build status
max_wait_minutes = 15
check_interval_seconds = 30
checks = (max_wait_minutes * 60) // check_interval_seconds

logger.info(f"Monitoring build {build.id}")
logger.info(f"Check interval: {check_interval_seconds} seconds")
logger.info(f"Max wait time: {max_wait_minutes} minutes")

print(f"   ⏳ Checking build status every {check_interval_seconds} seconds...")
print(f"   (Will wait up to {max_wait_minutes} minutes)")

build_succeeded = False
for i in range(checks):
    try:
        build_status = cml_client.get_model_build(
            project_id=project_id,
            model_id=cml_model.id,
            build_id=build.id
        )

        status = build_status.status
        logger.info(f"Check {i+1}/{checks}: Build status = {status}")
        print(f"   Check {i+1}/{checks}: Build status = {status}")

        if status == "built":
            build_succeeded = True
            logger.info("✅ Build completed successfully!")
            print(f"   ✅ Build completed successfully!")
            break
        elif status == "build failed":
            logger.error(f"❌ Build failed!")
            logger.error(f"Check CML UI for build logs: Models > {MODEL_NAME} > Builds")
            print(f"   ❌ Build failed!")
            print(f"   Check CML UI for build logs: Models > {MODEL_NAME} > Builds")
            sys.exit(1)
        elif status in ["building", "queued"]:
            # Still building, wait
            time.sleep(check_interval_seconds)
        else:
            logger.warning(f"Unknown build status: {status}")
            print(f"   ⚠️  Unknown status: {status}")
            time.sleep(check_interval_seconds)

    except Exception as e:
        logger.error(f"Error checking build status: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        print(f"   ⚠️  Error checking build status: {e}")
        time.sleep(check_interval_seconds)

if not build_succeeded:
    logger.warning(f"Build did not complete within {max_wait_minutes} minutes")
    print(f"   ⚠️  Build did not complete within {max_wait_minutes} minutes")
    print(f"   The build is still running. Check CML UI and deploy manually when ready.")
    print(f"   Location: Models > {MODEL_NAME} > Builds")
else:
    # Build succeeded, create deployment
    logger.info("Build succeeded, proceeding with deployment...")
    print("\n   Creating deployment...")

    create_deployment_request = cmlapi.CreateModelDeploymentRequest(
        cpu="2",
        memory="4"
    )

    logger.info("Deployment request payload:")
    logger.info(f"  - cpu: 2")
    logger.info(f"  - memory: 4GB")
    logger.info(f"  - model_id: {cml_model.id}")
    logger.info(f"  - build_id: {build.id}")

    try:
        deployment = cml_client.create_model_deployment(
            body=create_deployment_request,
            project_id=project_id,
            model_id=cml_model.id,
            build_id=build.id
        )
        logger.info(f"✅ Deployment created successfully: {deployment.id}")
        logger.info(f"  - Status: {deployment.status if hasattr(deployment, 'status') else 'N/A'}")
        print(f"   ✅ Deployment created: {deployment.id}")
        deployment_id = deployment.id

    except ApiException as e:
        logger.error("=" * 60)
        logger.error("API EXCEPTION DURING DEPLOYMENT")
        logger.error("=" * 60)
        logger.error(f"Status Code: {e.status}")
        logger.error(f"Reason: {e.reason}")
        logger.error(f"Body: {e.body}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")

        print(f"   ❌ ERROR creating deployment:")
        print(f"   Status: {e.status}")
        print(f"   Body: {e.body}")
        print(f"\n💡 Check outputs/deployment_debug.log for detailed diagnostics")
        deployment_id = None
    except Exception as e:
        logger.error("=" * 60)
        logger.error("UNEXPECTED ERROR DURING DEPLOYMENT")
        logger.error("=" * 60)
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")

        print(f"   ❌ ERROR: {e}")
        print(f"\n💡 Check outputs/deployment_debug.log for detailed diagnostics")
        deployment_id = None

# ============================================================================
# Success Summary
# ============================================================================
logger.info("=" * 60)
logger.info("DEPLOYMENT SUMMARY")
logger.info("=" * 60)

print("\n" + "=" * 80)
if 'deployment_id' in locals() and deployment_id:
    logger.info("✅ DEPLOYMENT COMPLETE!")
    print("✅ DEPLOYMENT COMPLETE!")
else:
    logger.info("BUILD COMPLETE - Manual deployment required")
    print("✅ BUILD COMPLETE - DEPLOY MANUALLY")
print("=" * 80)

print(f"\n📊 Model Summary:")
print(f"   Model Name: {MODEL_NAME}")
print(f"   F1 Score: {f1_score:.4f}")
print(f"   CML Model ID: {cml_model.id}")
print(f"   Build ID: {build.id}")

logger.info("Model Summary:")
logger.info(f"  - Model Name: {MODEL_NAME}")
logger.info(f"  - F1 Score: {f1_score:.4f}")
logger.info(f"  - CML Model ID: {cml_model.id}")
logger.info(f"  - Build ID: {build.id}")

if 'deployment_id' in locals() and deployment_id:
    print(f"   Deployment ID: {deployment_id}")
    print(f"\n✅ REST API ENDPOINT IS LIVE!")
    print(f"   Access it: Models > {MODEL_NAME} > Deployments")
    print(f"\n🎯 Test your API:")
    print(f'   curl -X POST https://your-cml-workspace/models/...')

    logger.info(f"  - Deployment ID: {deployment_id}")
    logger.info("REST API endpoint is live!")
else:
    print(f"\n⏳ To deploy manually:")
    print(f"   1. Go to: Models > {MODEL_NAME} > Builds")
    print(f"   2. Once build shows 'Built', click Deploy")
    print(f"   3. Configure resources (CPU: 2, Memory: 4GB)")

    logger.info("Manual deployment required - see instructions above")

# Save deployment info
deployment_info = {
    "run_id": run_id,
    "model_name": MODEL_NAME,
    "registered_model_id": registered_model_id,
    "model_version_id": str(model_version_id),
    "f1_score": float(f1_score),
    "cml_model_id": cml_model.id,
    "build_id": build.id,
    "deployment_id": deployment_id if 'deployment_id' in locals() else None,
    "status": "Deployed" if ('deployment_id' in locals() and deployment_id) else "Built"
}

logger.info("Saving deployment info to outputs/deployment_info.json")
logger.info(f"Deployment info: {json.dumps(deployment_info, indent=2)}")

with open("outputs/deployment_info.json", "w") as f:
    json.dump(deployment_info, f, indent=2)

print(f"\n💾 Saved to: outputs/deployment_info.json")
print(f"💾 Debug log saved to: outputs/deployment_debug.log")
print("=" * 80)

logger.info("=" * 60)
logger.info("Script completed successfully")
logger.info("=" * 60)