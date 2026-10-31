"""
Every lookout module imports with no secrets anywhere.

Each import runs in a fresh interpreter whose HOME and working directory are an
empty temp dir and whose env carries no secret names, so neither
`~/.config/lookout/secrets.toml` nor any `.streamlit/secrets.toml` can be
found. A module that reads a secret at import fails here.
"""

import os
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

import lookout
from lookout.secrets_loader import SECRET_NAMES

REPO_ROOT = Path(__file__).resolve().parent.parent

# Scratch notebooks whose third-party imports are not in requirements.txt.
NEEDS_UNLISTED_DEPS = {"lookout.cli.playground"}

MODULES = sorted(
    m.name
    for m in pkgutil.walk_packages(lookout.__path__, "lookout.")
    if m.name not in NEEDS_UNLISTED_DEPS
)


@pytest.fixture(scope="module")
def clean_env(tmp_path_factory):
    """An env with an empty HOME and no secret names set."""
    home = tmp_path_factory.mktemp("home")
    env = {k: v for k, v in os.environ.items() if k not in SECRET_NAMES}
    env["HOME"] = str(home)
    env["PYTHONPATH"] = str(REPO_ROOT)
    return env, home


def test_walk_finds_the_secret_reading_modules():
    for name in (
        "lookout.storage.storj",
        "lookout.api.ambient_client",
        "lookout.api.awn_controller",
        "lookout.cli.catchup",
    ):
        assert name in MODULES


def test_every_module_imports_with_no_secrets(clean_env):
    env, home = clean_env
    code = (
        "import importlib, sys\n"
        "failed = []\n"
        f"for name in {MODULES!r}:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except BaseException as e:\n"
        "        failed.append(f'{name}: {type(e).__name__}: {e}')\n"
        "print('\\n'.join(failed))\n"
        "sys.exit(1 if failed else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=home,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr[-2000:]
