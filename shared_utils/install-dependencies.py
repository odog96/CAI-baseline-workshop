#!/usr/bin/env python3
"""
AMP install task: install the workshop's Python dependencies.

Note on resources: this is a single blocking, single-threaded pip install of
packages that all ship prebuilt wheels, so nothing compiles and extra CPU cores
sit idle. .project-metadata.yaml requests cpu: 2 / memory: 8 for exactly that
reason -- a larger request only makes the pod harder for Kubernetes to schedule.
"""
import os
import site
import subprocess
import sys

REQUIREMENTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "requirements.txt")


def report_environment():
    """Print which interpreter is being used, and where packages will land.

    This is here because the install writes to ~/.local/lib/python3.<minor>/
    site-packages (pip.conf sets user = true), and that directory is not shared
    across Python minor versions. If a later job or model build runs on a
    different minor it will find no packages at all, and the failure shows up
    far away from this step. Logging the interpreter makes that diagnosable from
    the AMP log alone.
    """
    print(f"Python:            {sys.version.split()[0]} ({sys.executable})")
    try:
        print(f"User site-packages: {site.getusersitepackages()}")
    except Exception as error:          # pragma: no cover - informational only
        print(f"User site-packages: unavailable ({error})")
    print(f"Requirements:      {REQUIREMENTS}")
    print("-" * 70)


def install_requirements():
    """Install Python packages from requirements.txt"""
    try:
        print("Installing dependencies from requirements.txt...")
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-input",
            "--progress-bar", "off",
            "-r",
            REQUIREMENTS,
        ])
        print("✓ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing dependencies: {e}")
        sys.exit(1)


if __name__ == "__main__":
    report_environment()
    install_requirements()
