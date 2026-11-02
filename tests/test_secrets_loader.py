"""
Tests for lookout.secrets_loader: the one door every secret read goes through.

The loader looks in env vars, then the TOML secrets file, then st.secrets
while Streamlit is loaded. These tests drive it through `get_secret` only.
"""

import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from lookout import secrets_loader
from lookout.secrets_loader import MissingSecretError, get_secret, get_setting

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_file(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(0o600)
    return path


def test_env_var_is_read(no_secrets, monkeypatch):
    monkeypatch.setenv("AMBIENT_API_KEY", "from-env")
    assert get_secret("AMBIENT_API_KEY") == "from-env"


def test_top_level_name_read_from_file(no_secrets, monkeypatch):
    path = _write_file(
        no_secrets / "secrets.toml",
        'AMBIENT_API_KEY = "from-file"\nHC_PING_KEY = "ping"\n',
    )
    monkeypatch.setattr(secrets_loader, "SECRETS_FILE", path)
    assert get_secret("AMBIENT_API_KEY") == "from-file"
    assert get_secret("HC_PING_KEY") == "ping"


def test_storj_names_map_to_storage_options_table(no_secrets, monkeypatch):
    path = _write_file(
        no_secrets / "secrets.toml",
        "[lookout_storage_options]\n"
        'ACCESS_KEY_ID = "akid"\n'
        'SECRET_ACCESS_KEY = "sak"\n'
        'ENDPOINT_URL = "https://gateway.example"\n',
    )
    monkeypatch.setattr(secrets_loader, "SECRETS_FILE", path)
    assert get_secret("STORJ_ACCESS_KEY_ID") == "akid"
    assert get_secret("STORJ_SECRET_ACCESS_KEY") == "sak"
    assert get_secret("STORJ_ENDPOINT_URL") == "https://gateway.example"


def test_env_var_beats_file(no_secrets, monkeypatch):
    path = _write_file(no_secrets / "secrets.toml", 'AMBIENT_API_KEY = "file"\n')
    monkeypatch.setattr(secrets_loader, "SECRETS_FILE", path)
    monkeypatch.setenv("AMBIENT_API_KEY", "env")
    assert get_secret("AMBIENT_API_KEY") == "env"


def test_file_is_read_at_call_time_not_cached(no_secrets, monkeypatch):
    path = _write_file(no_secrets / "secrets.toml", 'AMBIENT_API_KEY = "one"\n')
    monkeypatch.setattr(secrets_loader, "SECRETS_FILE", path)
    assert get_secret("AMBIENT_API_KEY") == "one"
    _write_file(path, 'AMBIENT_API_KEY = "two"\n')
    assert get_secret("AMBIENT_API_KEY") == "two"


def test_streamlit_secrets_used_while_streamlit_is_loaded(no_secrets, monkeypatch):
    fake_st = types.ModuleType("streamlit")
    fake_st.secrets = {
        "AMBIENT_ENDPOINT": "https://st.example",
        "lookout_storage_options": {"ENDPOINT_URL": "https://st-storj.example"},
    }
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    assert get_secret("AMBIENT_ENDPOINT") == "https://st.example"
    assert get_secret("STORJ_ENDPOINT_URL") == "https://st-storj.example"


def test_file_beats_streamlit_secrets(no_secrets, monkeypatch):
    path = _write_file(no_secrets / "secrets.toml", 'AMBIENT_ENDPOINT = "file"\n')
    monkeypatch.setattr(secrets_loader, "SECRETS_FILE", path)
    fake_st = types.ModuleType("streamlit")
    fake_st.secrets = {"AMBIENT_ENDPOINT": "streamlit"}
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    assert get_secret("AMBIENT_ENDPOINT") == "file"


def test_streamlit_without_secrets_file_counts_as_missing(no_secrets, monkeypatch):
    class RaisingSecrets:
        def __getitem__(self, key):
            raise FileNotFoundError("No secrets found")

    fake_st = types.ModuleType("streamlit")
    fake_st.secrets = RaisingSecrets()
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    with pytest.raises(MissingSecretError):
        get_secret("AMBIENT_API_KEY")


def test_missing_secret_names_the_secret_and_the_file(no_secrets):
    with pytest.raises(MissingSecretError) as exc:
        get_secret("HC_PING_KEY")
    message = str(exc.value)
    assert "HC_PING_KEY" in message
    assert "absent.toml" in message


def test_missing_secret_is_a_key_error(no_secrets):
    with pytest.raises(KeyError):
        get_secret("AMBIENT_API_KEY")


def test_unknown_name_is_rejected(no_secrets, monkeypatch):
    monkeypatch.setenv("NOT_A_LOOKOUT_SECRET", "x")
    with pytest.raises(ValueError):
        get_secret("NOT_A_LOOKOUT_SECRET")


def test_loose_file_permissions_log_a_warning(no_secrets, monkeypatch, caplog):
    path = no_secrets / "secrets.toml"
    path.write_text('AMBIENT_API_KEY = "file"\n')
    path.chmod(0o644)
    monkeypatch.setattr(secrets_loader, "SECRETS_FILE", path)
    caplog.set_level("WARNING")
    assert get_secret("AMBIENT_API_KEY") == "file"
    assert "chmod 600" in caplog.text


def test_loader_does_not_import_streamlit():
    code = (
        "import sys; from lookout.secrets_loader import get_secret; "
        "assert 'streamlit' not in sys.modules, 'streamlit imported'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_example_file_lists_every_name_with_no_values():
    import tomllib

    example = tomllib.loads((REPO_ROOT / "secrets.example.toml").read_text())
    storage = example.pop("lookout_storage_options")
    assert storage == {
        "ACCESS_KEY_ID": "",
        "SECRET_ACCESS_KEY": "",
        "ENDPOINT_URL": "",
    }
    assert example == {
        "AMBIENT_API_KEY": "",
        "AMBIENT_APPLICATION_KEY": "",
        "AMBIENT_ENDPOINT": "",
        "HC_PING_KEY": "",
        "STATION_MAC": "",
    }


def test_secret_names_are_the_spec_names():
    assert set(secrets_loader.SECRET_NAMES) == {
        "AMBIENT_API_KEY",
        "AMBIENT_APPLICATION_KEY",
        "AMBIENT_ENDPOINT",
        "HC_PING_KEY",
        "STORJ_ACCESS_KEY_ID",
        "STORJ_SECRET_ACCESS_KEY",
        "STORJ_ENDPOINT_URL",
    }


def test_default_file_is_under_user_config():
    assert secrets_loader.DEFAULT_SECRETS_FILE == (
        Path(os.path.expanduser("~")) / ".config" / "lookout" / "secrets.toml"
    )


# ---------- settings: same sources, but optional ----------


def test_setting_falls_back_to_its_default_when_unset(no_secrets, monkeypatch):
    monkeypatch.delenv("STATION_MAC", raising=False)
    assert get_setting("STATION_MAC") is None
    assert get_setting("STATION_MAC", "fallback") == "fallback"


def test_setting_read_from_env_then_file(no_secrets, monkeypatch):
    monkeypatch.delenv("STATION_MAC", raising=False)
    path = _write_file(no_secrets / "secrets.toml", 'STATION_MAC = "AA:01"\n')
    monkeypatch.setattr(secrets_loader, "SECRETS_FILE", path)
    assert get_setting("STATION_MAC", "fallback") == "AA:01"
    monkeypatch.setenv("STATION_MAC", "AA:02")
    assert get_setting("STATION_MAC", "fallback") == "AA:02"


def test_setting_read_from_streamlit_secrets(no_secrets, monkeypatch):
    monkeypatch.delenv("STATION_MAC", raising=False)
    fake_st = types.ModuleType("streamlit")
    fake_st.secrets = {"STATION_MAC": "AA:03"}
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    assert get_setting("STATION_MAC") == "AA:03"


def test_unknown_setting_name_is_rejected():
    with pytest.raises(ValueError):
        get_setting("STATION_NAME")


def test_dev_mode_setting_reads_streamlits_environment_table(no_secrets, monkeypatch):
    monkeypatch.delenv("DEV_MODE", raising=False)
    fake_st = types.ModuleType("streamlit")
    fake_st.secrets = {"environment": {"dev": True}}
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    assert get_setting("DEV_MODE") == "True"


def test_settings_are_not_secret_names():
    assert secrets_loader.SETTING_NAMES == ("STATION_MAC", "DEV_MODE")
    assert not set(secrets_loader.SETTING_NAMES) & set(secrets_loader.SECRET_NAMES)
