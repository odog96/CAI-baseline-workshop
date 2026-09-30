"""
ML Runtime lookup for Cloudera AI
Runtime image tags differ per workspace, so resolve them at run time
"""

import json

# All workshop code, jobs and model builds run on this Python version
PYTHON_VERSION = "3.12"


def get_workbench_runtime(client, python_version=PYTHON_VERSION):
    """
    Return the image identifier of the newest enabled PBJ Workbench Standard
    runtime, preferring the given Python version (default: PYTHON_VERSION).

    Args:
        client: cmlapi client
        python_version: e.g. "3.12"; falls back to any Python if not available

    Returns:
        Runtime image identifier string
    """
    runtimes = client.list_runtimes(
        search_filter=json.dumps({"editor": "PBJ Workbench", "edition": "Standard"}),
        page_size=500
    ).runtimes
    runtimes = [r for r in runtimes if r.status == "ENABLED"]
    matching = [r for r in runtimes if r.kernel == f"Python {python_version}"] or runtimes
    if not matching:
        raise RuntimeError("No enabled PBJ Workbench Standard runtime found in this workspace")

    return max(matching, key=lambda r: r.image_identifier.rsplit(":", 1)[-1]).image_identifier
