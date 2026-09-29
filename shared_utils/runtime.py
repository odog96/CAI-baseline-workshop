"""
ML Runtime discovery for Cloudera AI.

Why this exists: jobs and model builds need a runtime image identifier. Writing
one in as a literal string breaks every time the workspace is upgraded or the
project moves to a different environment, because image identifiers embed the
registry host, the Python minor version and the build tag -- all three of which
differ between public cloud and on-prem.

The rule that matters: the dependency install writes to
~/.local/lib/python3.<minor>/site-packages, which is NOT shared across Python
minor versions and holds ABI-specific wheels. So a job scheduled on a different
minor than the one that ran the install starts with no pandas, no mlflow, no
onnxruntime -- and fails a long way from the real cause. That is why
resolve_runtime() defaults the kernel to the *running* interpreter and never
relaxes it, even though it will happily relax the editor.
"""

import json
import sys

import cmlapi


def current_kernel():
    """The kernel string matching the interpreter executing this code."""
    return f"Python {sys.version_info.major}.{sys.version_info.minor}"


def list_available_runtimes(client=None):
    """(editor, kernel, edition, full_version) for every runtime in the catalog."""
    client = client or cmlapi.default_client()
    return sorted({
        (r.editor, r.kernel, r.edition, r.full_version)
        for r in client.list_runtimes().runtimes
    })


def resolve_runtime(kernel=None, editor="PBJ Workbench", edition="Standard",
                    client=None):
    """
    Return the image identifier of the newest runtime matching the criteria.

    Args:
        kernel: e.g. "Python 3.12". Defaults to the running interpreter, which
            is almost always what you want -- see the module docstring.
        editor: preferred editor. Relaxed automatically if it has no match,
            since the editor only affects the interactive UI, not the ABI.
        edition: "Standard" or "Nvidia GPU".

    Raises:
        RuntimeError: listing what the catalog actually contains, so the fix is
            obvious instead of requiring a separate investigation.
    """
    kernel = kernel or current_kernel()
    client = client or cmlapi.default_client()

    # Try the requested editor first, then fall back. Never relax `kernel`.
    candidates = [editor] + [e for e in ("PBJ Workbench", "JupyterLab") if e != editor]
    for candidate in candidates:
        runtimes = client.list_runtimes(search_filter=json.dumps({
            "kernel": kernel,
            "edition": edition,
            "editor": candidate,
        })).runtimes
        if runtimes:
            newest = sorted(runtimes, key=lambda r: r.full_version)[-1]
            return newest.image_identifier

    available = "\n".join(f"  {t}" for t in list_available_runtimes(client))
    raise RuntimeError(
        f"No ML Runtime found for kernel={kernel!r} edition={edition!r} "
        f"(tried editors: {', '.join(candidates)}).\n"
        f"Available runtimes on this workspace:\n{available}"
    )
