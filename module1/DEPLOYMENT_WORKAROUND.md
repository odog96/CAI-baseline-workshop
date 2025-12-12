# Model Deployment Workaround Guide

## Problem Summary
The script `04_deploy.py` is failing with a **401 Unauthorized** error when trying to register models programmatically via the CML API. This is a permissions issue where the user account (`ozarate`) lacks the `CREATEMODEL` permission.

## Root Cause
```
Error: "failed to CreateRegisteredModel err: 401 - user Unauthorized to perform operation CREATEMODEL"
```

Both the CML API (`create_registered_model()`) and MLflow's `register_model()` use the same underlying CML API endpoint, so both hit the same permission restriction.

## Solution: Manual Registration Workaround

Since you mentioned you **can register models via the UI**, here's how to work around the permission issue:

### Step 1: Manually Register the Model via UI

1. Go to CML UI → **Models** → **Model Registry**
2. Click **"Register Model"**
3. Fill in the details:
   - **Model Name**: `banking_campaign_predictor`
   - **Experiment**: `BANK_MARKETING_EXPERIMENTS_ozarate`
   - **Run ID**: Get this from the script output (it's printed in Step 1)
   - **Model Path**: `model`
4. Click **Register**
5. Note down the IDs that are created:
   - **Registered Model ID** (format: `xxxx-xxxx-xxxx-xxxx`)
   - **Model Version ID** (format: `xxxx-xxxx-xxxx-xxxx`)

### Step 2: Get the Model IDs

After registering via UI, you need to get the two IDs. You can find them in:
- The URL when viewing the model in the registry
- Or run this Python snippet in a CML session:

```python
import cmlapi

cml_client = cmlapi.default_client()
models = cml_client.list_registered_models()

MODEL_NAME = "banking_campaign_predictor"
for model in models.models:
    if model.name == MODEL_NAME:
        print(f"Registered Model ID: {model.model_id}")

        # Get versions
        versions = cml_client.list_model_versions(
            registered_model_id=model.model_id
        )
        if versions.model_versions:
            print(f"Model Version ID: {versions.model_versions[0].model_version_id}")
        break
```

### Step 3: Run the Script with Manual IDs

Once you have the IDs, export them as environment variables and run the script:

```bash
export REGISTERED_MODEL_ID="your-registered-model-id-here"
export MODEL_VERSION_ID="your-model-version-id-here"
python 04_deploy.py
```

The script will detect these environment variables and skip the registration step, proceeding directly to build and deployment.

## Alternative: Fix the Permissions

If you want to fix the underlying permission issue, contact your CML administrator and request:

1. **CREATEMODEL** permission for your user account
2. Or elevated role (e.g., "Business User" or "Admin" role in the project)
3. Check if Model Registry is enabled for your project

## What Was Improved in the Script

The updated `04_deploy.py` now includes:

✅ **Comprehensive logging** - All actions logged to `outputs/deployment_debug.log`
✅ **Detailed error messages** - Full stack traces, request payloads, and response details
✅ **Permission diagnostics** - Detects 401 errors and provides troubleshooting steps
✅ **Manual ID support** - Can use `REGISTERED_MODEL_ID` and `MODEL_VERSION_ID` env vars
✅ **Better error handling** - Catches and logs all API exceptions with context
✅ **Request logging** - Logs every API call before execution for debugging

## Files Created

- `outputs/deployment_debug.log` - Detailed debug log with all API calls and errors
- `outputs/deployment_info.json` - Deployment metadata (created on success)

## Next Steps

1. Try the manual registration workaround above
2. If successful, the script will continue with:
   - Creating the CML model
   - Building the model container
   - Deploying the REST API endpoint
3. Check the debug log for any issues: `cat outputs/deployment_debug.log`
