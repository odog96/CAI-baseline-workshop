"""
Helper script to get Registered Model ID and Model Version ID
after manually registering a model via the CML UI.

Usage:
    python get_model_ids.py

This will list all registered models you have access to and show their IDs.
"""

import cmlapi
import os

MODEL_NAME = "banking_campaign_predictor"
USERNAME = os.environ.get('HADOOP_USER_NAME') or os.environ.get("PROJECT_OWNER")

print("=" * 80)
print("Registered Model ID Finder")
print("=" * 80)
print(f"\nSearching for model: {MODEL_NAME}")
print(f"User: {USERNAME}\n")

try:
    cml_client = cmlapi.default_client()
    project_id = os.environ.get("CDSW_PROJECT_ID")

    # List all registered models
    models = cml_client.list_registered_models()

    if not hasattr(models, 'models') or not models.models:
        print("❌ No registered models found")
        print("\n💡 Have you registered the model via the UI yet?")
        print("   Go to: Models > Model Registry > Register Model")
        exit(1)

    print(f"Found {len(models.models)} total registered models\n")

    # Filter to this project
    project_models = []
    other_models = []

    for model in models.models:
        if hasattr(model, 'project_id') and model.project_id == project_id:
            project_models.append(model)
        else:
            other_models.append(model)

    print(f"📊 Models in this project: {len(project_models)}")
    print(f"📊 Models in other projects: {len(other_models)}\n")

    # Find our specific model
    target_model = None
    for model in project_models:
        if model.name == MODEL_NAME:
            target_model = model
            break

    if target_model:
        print("=" * 80)
        print(f"✅ FOUND MODEL: {MODEL_NAME}")
        print("=" * 80)
        print(f"\n📝 Registered Model ID: {target_model.model_id}")
        print(f"   Project ID: {target_model.project_id if hasattr(target_model, 'project_id') else 'N/A'}")
        print(f"   Created: {target_model.created_at if hasattr(target_model, 'created_at') else 'N/A'}")

        # Get versions
        try:
            versions = cml_client.list_model_versions(
                registered_model_id=target_model.model_id
            )

            if hasattr(versions, 'model_versions') and versions.model_versions:
                print(f"\n📦 Model Versions ({len(versions.model_versions)} total):")
                for i, version in enumerate(versions.model_versions):
                    print(f"\n   Version {i + 1}:")
                    print(f"   📝 Model Version ID: {version.model_version_id}")
                    print(f"      Status: {version.status if hasattr(version, 'status') else 'N/A'}")
                    print(f"      Created: {version.created_at if hasattr(version, 'created_at') else 'N/A'}")

                # Show the export command for the latest version
                latest_version = versions.model_versions[0]
                print("\n" + "=" * 80)
                print("✅ TO USE THIS MODEL, RUN:")
                print("=" * 80)
                print(f"\nexport REGISTERED_MODEL_ID=\"{target_model.model_id}\"")
                print(f"export MODEL_VERSION_ID=\"{latest_version.model_version_id}\"")
                print(f"python 04_deploy.py")
                print("\n" + "=" * 80)

            else:
                print("\n⚠️  No versions found for this model")
                print("   You may need to create a version via the UI")

        except Exception as e:
            print(f"\n❌ Error getting versions: {str(e)}")

    else:
        print(f"❌ Model '{MODEL_NAME}' not found in this project")
        print("\nAvailable models in this project:")
        if project_models:
            for model in project_models:
                print(f"   - {model.name} (ID: {model.model_id})")
        else:
            print("   (none)")

        print(f"\n💡 To register '{MODEL_NAME}':")
        print("   1. Go to: Models > Model Registry > Register Model")
        print("   2. Use these details:")
        print(f"      - Model Name: {MODEL_NAME}")
        print(f"      - Experiment: BANK_MARKETING_EXPERIMENTS_{USERNAME}")
        print("      - Run ID: (get from 04_deploy.py output)")
        print("      - Model Path: model")

except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    print(f"\nFull traceback:\n{traceback.format_exc()}")

print("\n" + "=" * 80)
