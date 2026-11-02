"""
secrets_loader.py: The one place Lookout reads a secret.

`get_secret(name)` takes a flat secret name and looks in, in order:

1. the environment variable of that name;
2. the TOML secrets file, `~/.config/lookout/secrets.toml` (mode 600, outside
   the git checkout), read with stdlib `tomllib`.

Nothing is read until `get_secret` is called, so importing a module that needs
secrets never fails for lack of them.

The TOML file keeps the shape of the old Streamlit secrets file: the Ambient
and Healthchecks names are top-level keys, and the three Storj names map to
keys in the `[lookout_storage_options]` table. `secrets.example.toml` at the
repo root lists every name with no values.

Non-secret settings (`SETTING_NAMES`: `STATION_MAC`, `DEV_MODE`) come from the
same sources in the same order, through `get_setting(name, default)`.

Functions:
- get_secret(name): Return the value of one secret, or raise MissingSecretError.
- get_setting(name, default): Return one setting, or the default when unset.

Example:
    from lookout.secrets_loader import get_secret
    api_key = get_secret("AMBIENT_API_KEY")
"""

import os
import stat
import tomllib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from lookout.utils.log_util import app_logger

logger = app_logger(__name__)

DEFAULT_SECRETS_FILE = Path.home() / ".config" / "lookout" / "secrets.toml"

# Read at call time, so tests and callers can point it elsewhere.
SECRETS_FILE = DEFAULT_SECRETS_FILE

STORAGE_TABLE = "lookout_storage_options"

# Flat name -> path of keys inside the TOML file.
SECRET_PATHS: Dict[str, Tuple[str, ...]] = {
    "AMBIENT_API_KEY": ("AMBIENT_API_KEY",),
    "AMBIENT_APPLICATION_KEY": ("AMBIENT_APPLICATION_KEY",),
    "AMBIENT_ENDPOINT": ("AMBIENT_ENDPOINT",),
    "HC_PING_KEY": ("HC_PING_KEY",),
    "STORJ_ACCESS_KEY_ID": (STORAGE_TABLE, "ACCESS_KEY_ID"),
    "STORJ_SECRET_ACCESS_KEY": (STORAGE_TABLE, "SECRET_ACCESS_KEY"),
    "STORJ_ENDPOINT_URL": (STORAGE_TABLE, "ENDPOINT_URL"),
}

SECRET_NAMES: Tuple[str, ...] = tuple(SECRET_PATHS)

# Non-secret settings, read from the same sources as the secrets. Kept apart
# from SECRET_NAMES because a setting may be unset (its reader has a default).
SETTING_PATHS: Dict[str, Tuple[str, ...]] = {
    "STATION_MAC": ("STATION_MAC",),
    # `[environment] dev = true` in the file, or the DEV_MODE env var.
    "DEV_MODE": ("environment", "dev"),
}

SETTING_NAMES: Tuple[str, ...] = tuple(SETTING_PATHS)


class MissingSecretError(KeyError):
    """Raised when no source holds the requested secret."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


def _lookup(source: Any, path: Tuple[str, ...]) -> Optional[str]:
    """
    Walk a key path through a mapping-like source.

    :param source: A dict parsed from the secrets file.
    :param path: Keys to follow, outermost first.
    :return: The value as a string, or None when any key is absent or empty.
    """
    node = source
    for key in path:
        try:
            node = node[key]
        except (KeyError, TypeError):
            return None
    if node is None or node == "":
        return None
    return str(node)


def _read_file(path: Path) -> Dict[str, Any]:
    """
    Parse the TOML secrets file, warning when others can read it.

    :param path: Location of the secrets file.
    :return: Parsed contents, or an empty dict when the file is absent.
    """
    if not path.is_file():
        return {}
    mode = path.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        logger.warning(f"Secrets file {path} is readable by others; chmod 600 it")
    with path.open("rb") as f:
        return tomllib.load(f)


def _resolve(name: str, path: Tuple[str, ...]) -> Optional[str]:
    """
    Look one name up in the env var, then the secrets file.

    :param name: The flat name, also the env var's name.
    :param path: Keys to follow inside the file.
    :return: The first value found, or None.
    """
    value = os.environ.get(name)
    if value:
        return value

    return _lookup(_read_file(SECRETS_FILE), path)


def get_secret(name: str) -> str:
    """
    Return one secret, from the env var, then the secrets file.

    :param name: A flat secret name from SECRET_NAMES, e.g. "AMBIENT_API_KEY".
    :return: The secret's value.
    :raises ValueError: If the name is not a known Lookout secret.
    :raises MissingSecretError: If no source holds a value for it.
    """
    if name not in SECRET_PATHS:
        raise ValueError(f"Unknown secret name {name!r}; known: {SECRET_NAMES}")

    path = SECRET_PATHS[name]
    value = _resolve(name, path)
    if value is not None:
        return value

    raise MissingSecretError(
        f"Secret {name} not found: set the {name} env var or add "
        f"{'.'.join(path)} to {SECRETS_FILE}"
    )


def get_setting(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Return one setting, from the same sources and in the same order as secrets.

    :param name: A flat setting name from SETTING_NAMES, e.g. "STATION_MAC".
    :param default: Returned when no source holds a value.
    :return: The setting's value, or `default`.
    :raises ValueError: If the name is not a known Lookout setting.
    """
    if name not in SETTING_PATHS:
        raise ValueError(f"Unknown setting name {name!r}; known: {SETTING_NAMES}")

    value = _resolve(name, SETTING_PATHS[name])
    return default if value is None else value
