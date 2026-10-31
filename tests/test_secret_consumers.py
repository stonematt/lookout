"""
The Storj and Ambient clients read their secrets when called, not at import.

Each test sets a secret after the module is imported and checks the value
reaches the outgoing client call, at the boundary (boto3, requests).
"""

import pytest

import lookout.api.ambient_client as ambient_client
import lookout.storage.storj as storj
from lookout.secrets_loader import MissingSecretError


class FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture
def captured_get(monkeypatch):
    calls = []

    def fake_get(url, params=None):
        calls.append((url, params))
        return FakeResponse([{"macAddress": "AA:BB"}])

    monkeypatch.setattr(ambient_client.requests, "get", fake_get)
    monkeypatch.setattr(ambient_client.time, "sleep", lambda s: None)
    return calls


def test_s3_client_uses_storj_secrets_set_after_import(monkeypatch):
    monkeypatch.setenv("STORJ_ACCESS_KEY_ID", "akid-late")
    monkeypatch.setenv("STORJ_SECRET_ACCESS_KEY", "sak-late")
    monkeypatch.setenv("STORJ_ENDPOINT_URL", "https://storj.test")
    seen = {}
    monkeypatch.setattr(storj.boto3, "client", lambda *a, **kw: seen.update(kw))

    storj.get_s3_client()

    assert seen["endpoint_url"] == "https://storj.test"
    assert seen["aws_access_key_id"] == "akid-late"
    assert seen["aws_secret_access_key"] == "sak-late"


def test_s3_client_without_secrets_raises_missing_secret(monkeypatch, no_secrets):
    monkeypatch.setattr(storj.boto3, "client", lambda *a, **kw: None)
    with pytest.raises(MissingSecretError):
        storj.get_s3_client()


def test_get_devices_uses_ambient_secrets_set_after_import(monkeypatch, captured_get):
    monkeypatch.setenv("AMBIENT_ENDPOINT", "https://ambient.test/v1/")
    monkeypatch.setenv("AMBIENT_API_KEY", "api-late")
    monkeypatch.setenv("AMBIENT_APPLICATION_KEY", "app-late")

    assert ambient_client.get_devices() == [{"macAddress": "AA:BB"}]

    url, params = captured_get[0]
    assert url == "https://ambient.test/v1/devices"
    assert params == {"apiKey": "api-late", "applicationKey": "app-late"}


def test_device_history_request_carries_keys_and_limit(monkeypatch, captured_get):
    monkeypatch.setenv("AMBIENT_ENDPOINT", "https://ambient.test/v1")
    monkeypatch.setenv("AMBIENT_API_KEY", "api")
    monkeypatch.setenv("AMBIENT_APPLICATION_KEY", "app")

    ambient_client.get_device_history_raw("AA:BB", limit=5, end_date=1700000000000)

    url, params = captured_get[0]
    assert url == "https://ambient.test/v1/devices/AA:BB"
    assert params == {
        "apiKey": "api",
        "applicationKey": "app",
        "limit": 5,
        "endDate": 1700000000000,
    }
