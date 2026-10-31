"""
Shared pytest fixtures. The suite is hermetic: no test sees a real secret.

`stub_secrets` runs for every test. It sets each name in
`lookout.secrets_loader.SECRET_NAMES` to a fake env value and points the
loader's secrets file at a path that does not exist, so a developer's
`~/.config/lookout/secrets.toml` or `.streamlit/secrets.toml` is never read
through `get_secret`. Add new fixtures below; keep this one autouse.
"""

import sys

import pytest

from lookout import secrets_loader

FAKE_SECRETS = {
    "AMBIENT_API_KEY": "test-ambient-api-key",
    "AMBIENT_APPLICATION_KEY": "test-ambient-application-key",
    "AMBIENT_ENDPOINT": "https://ambient.invalid/v1",
    "HC_PING_KEY": "test-hc-ping-key",
    "STORJ_ACCESS_KEY_ID": "test-storj-access-key-id",
    "STORJ_SECRET_ACCESS_KEY": "test-storj-secret-access-key",
    "STORJ_ENDPOINT_URL": "https://storj.invalid",
}


@pytest.fixture(autouse=True)
def stub_secrets(monkeypatch, tmp_path_factory):
    """
    Give every test fake secrets and no secrets file.

    :return: The fake values, keyed by secret name.
    """
    missing_file = tmp_path_factory.mktemp("secrets") / "secrets.toml"
    monkeypatch.setattr(secrets_loader, "SECRETS_FILE", missing_file)
    for name in secrets_loader.SECRET_NAMES:
        monkeypatch.setenv(name, FAKE_SECRETS[name])
    return dict(FAKE_SECRETS)


@pytest.fixture
def no_secrets(monkeypatch, tmp_path):
    """
    Remove every secret source: env vars, the secrets file and st.secrets.

    :return: A temp dir the test may write a secrets file into.
    """
    for name in secrets_loader.SECRET_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(secrets_loader, "SECRETS_FILE", tmp_path / "absent.toml")
    monkeypatch.delitem(sys.modules, "streamlit", raising=False)
    return tmp_path
