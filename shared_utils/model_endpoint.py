"""
Resolve a deployed model's endpoint and access key -- WITHOUT storing either in
source control.

Background: this file replaces a hardcoded endpoint URL and access key that were
committed to a public GitHub repository. A model access key authorises calls to
the model's REST endpoint, so committing one publishes a working credential, and
deleting the line later does not help -- git history keeps it. The lesson worth
carrying forward is that the credential should never be in a file that git can
see in the first place.

Resolution order, most explicit first:
  1. MODEL_ENDPOINT / MODEL_ACCESS_KEY environment variables.
  2. A local, git-ignored JSON file (.secrets/model_credentials.json).
  3. The CML API, looked up live by model name. This is the preferred path for
     the lab: each participant deploys their own model, and the key is fetched
     at runtime from the workspace that issued it, so there is no secret at rest
     anywhere and nothing to rotate or leak.
"""

import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CREDENTIALS_FILE = os.path.join(PROJECT_ROOT, ".secrets", "model_credentials.json")


def _from_environment():
    endpoint = os.environ.get("MODEL_ENDPOINT", "").strip()
    access_key = os.environ.get("MODEL_ACCESS_KEY", "").strip()
    if endpoint and access_key:
        return {"model_endpoint": endpoint, "access_key": access_key,
                "source": "environment variables"}
    return None


def _from_file():
    if not os.path.exists(CREDENTIALS_FILE):
        return None
    try:
        with open(CREDENTIALS_FILE) as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    endpoint = str(data.get("model_endpoint", "")).strip()
    access_key = str(data.get("access_key", "")).strip()
    if endpoint and access_key:
        return {"model_endpoint": endpoint, "access_key": access_key,
                "source": CREDENTIALS_FILE}
    return None


def _from_cml_api(model_name):
    """Look the model up in this workspace and read its access key at runtime."""
    try:
        import cmlapi
    except ImportError:
        return None

    try:
        client = cmlapi.default_client()
        project_id = os.environ["CDSW_PROJECT_ID"]
        models = client.list_models(project_id=project_id).models or []
    except Exception:
        return None

    matches = [m for m in models if m.name == model_name] or models
    if not matches:
        return None

    model = matches[0]
    access_key = getattr(model, "access_key", None)
    if not access_key:
        return None

    # Build the endpoint from this workspace's own domain, so it is correct on
    # prem and on public cloud without anyone editing a URL.
    domain = os.environ.get("CDSW_DOMAIN", "")
    if not domain:
        return None

    return {
        "model_endpoint": f"https://modelservice.{domain}/model",
        "access_key": access_key,
        "source": f"CML API (model {model.name!r})",
    }


def get_model_endpoint_config(model_name=None, required=True):
    """
    Return {"model_endpoint": ..., "access_key": ...} for the deployed model.

    Args:
        model_name: model to look up via the API. Defaults to API_CONFIG's name.
        required: raise if nothing resolves. Set False to get None instead.
    """
    if model_name is None:
        from .config import API_CONFIG
        model_name = API_CONFIG.get("model_name")

    for resolver in (_from_environment, _from_file,
                     lambda: _from_cml_api(model_name)):
        config = resolver()
        if config:
            # Never print the key itself -- only where it came from.
            print(f"✓ Model endpoint resolved from {config.pop('source')}")
            return config

    if not required:
        return None

    raise RuntimeError(
        "Could not resolve the model endpoint and access key.\n"
        "This is deliberately not stored in source control -- a committed access "
        "key is a published credential.\n"
        "Provide it in one of these ways:\n"
        "  1. Deploy the model first (module1/04_deploy.py); the key is then read\n"
        "     from the CML API automatically -- nothing to configure.\n"
        "  2. Set MODEL_ENDPOINT and MODEL_ACCESS_KEY in Project Settings > Advanced.\n"
        f"  3. Create {CREDENTIALS_FILE} with keys 'model_endpoint' and\n"
        "     'access_key'. That path is git-ignored.\n"
        "Find both values in the Cloudera AI UI under Model Deployments > your "
        "model > Settings."
    )
