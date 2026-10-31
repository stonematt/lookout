"""
Shared pytest fixtures. The suite is hermetic: no test sees a real secret.

`stub_secrets` runs for every test. It sets each name in
`lookout.secrets_loader.SECRET_NAMES` to a fake env value and points the
loader's secrets file at a path that does not exist, so a developer's
`~/.config/lookout/secrets.toml` or `.streamlit/secrets.toml` is never read
through `get_secret`. Add new fixtures below; keep this one autouse.

`fake_s3` serves the Storj gateway from moto, with the `lookout` bucket made;
`frozen_today` fixes the date `storj.backup_data` stamps its folders with.
`ambient_replay` answers the Ambient client from recorded responses. The
archive builders and loaders they pair with live in `tests/fixtures/`.
"""

import sys
from typing import Iterator, List, Optional
from types import SimpleNamespace

import pytest
from moto import mock_aws

from lookout import secrets_loader
from tests.fixtures.ambient import AmbientReplay
from tests.fixtures.storage import BUCKET, FROZEN_DAY, FrozenDatetime

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


@pytest.fixture
def fake_s3(monkeypatch: pytest.MonkeyPatch) -> Iterator[object]:
    """
    Fake the Storj gateway with moto and create the `lookout` bucket.

    `lookout.storage.storj.get_s3_client` runs unchanged against the stubbed
    endpoint, so every caller that builds its own client lands here too.

    :return: An S3 client on the fake gateway.
    """
    from lookout.storage import storj

    monkeypatch.setenv("MOTO_S3_CUSTOM_ENDPOINTS", FAKE_SECRETS["STORJ_ENDPOINT_URL"])
    with mock_aws():
        client = storj.get_s3_client()
        client.create_bucket(Bucket=BUCKET)
        yield client


@pytest.fixture
def ambient_replay(monkeypatch: pytest.MonkeyPatch) -> AmbientReplay:
    """
    Answer `lookout.api.ambient_client` from recorded responses, with no sleeps.

    Only the client module's `requests` and `time` are replaced; callers that
    sleep on their own (e.g. `awn_controller`) still do.

    Replace `.devices` or `.history` on the returned object to serve others.

    :return: The `AmbientReplay`; its `.calls` lists each request made.
    """
    from lookout.api import ambient_client

    replay = AmbientReplay()
    # Swap the module's own names only, so nothing else loses requests or sleep.
    monkeypatch.setattr(ambient_client, "requests", SimpleNamespace(get=replay))
    monkeypatch.setattr(ambient_client, "time", SimpleNamespace(sleep=lambda _s: None))
    return replay


@pytest.fixture
def frozen_today(monkeypatch: pytest.MonkeyPatch) -> str:
    """
    Fix the date `lookout.storage.storj.backup_data` names its folder with.

    :return: That date as `YYYYMMDD`.
    """
    from lookout.storage import storj

    monkeypatch.setattr(storj, "datetime", SimpleNamespace(datetime=FrozenDatetime))
    return FROZEN_DAY


class FakeHealthchecks:
    """
    A stand-in for the Healthchecks ping endpoint.

    Attributes: `urls` lists each full ping URL, `pings` each as
    `<slug>[/<suffix>]`; set `fail` to True and every ping raises, as if
    Healthchecks were down.
    """

    def __init__(self) -> None:
        self.pings: List[str] = []
        self.urls: List[str] = []
        self.fail = False

    def __call__(
        self, url: str, params: Optional[dict] = None, timeout: Optional[int] = None
    ) -> SimpleNamespace:
        if self.fail:
            raise ConnectionError("healthchecks is down")
        self.urls.append(url)
        self.pings.append(url.split("/", 4)[4])
        return SimpleNamespace(raise_for_status=lambda: None)


@pytest.fixture(autouse=True)
def healthchecks(monkeypatch: pytest.MonkeyPatch) -> FakeHealthchecks:
    """
    Answer `lookout.utils.healthchecks` from a fake, so no test pings the web.

    Only the module's own `requests` name is replaced. `.pings` lists each
    ping as `<slug>[/<suffix>]`; set `.fail = True` to simulate an outage.

    :return: The `FakeHealthchecks`.
    """
    from lookout.utils import healthchecks as hc

    fake = FakeHealthchecks()
    monkeypatch.setattr(hc, "requests", SimpleNamespace(get=fake))
    return fake
