"""Check all registered models across all projects"""
import cmlapi

cml_client = cmlapi.default_client()
models = cml_client.list_registered_models()

print(f"Total registered models: {len(models.models)}\n")

for i, model in enumerate(models.models, 1):
    print(f"{i}. {model.name}")
    print(f"   ID: {model.model_id}")
    print(f"   Project ID: {model.project_id if hasattr(model, 'project_id') else 'N/A'}")
    print(f"   Created: {model.created_at if hasattr(model, 'created_at') else 'N/A'}")

    # Check for our model
    if 'banking' in model.name.lower() or 'campaign' in model.name.lower():
        print(f"   ⭐ POSSIBLE MATCH!")
    print()
