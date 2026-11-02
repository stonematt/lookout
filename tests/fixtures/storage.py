"""
storage.py: Helpers for tests that run against the `fake_s3` fixture.

`BUCKET` is the bucket `fake_s3` creates. `bucket_keys(client)` lists it.
`FlakyStorj` and `flaky_storj` make chosen storage calls fail like an outage.
`FrozenDatetime` stands in for `datetime.datetime` with a fixed `now()`; the
`frozen_today` fixture in conftest installs it where `backup_data` reads it.
"""

import datetime as real_datetime
from typing import Dict, List, Optional

import pytest

from lookout.storage import storj

BUCKET = "lookout"
FROZEN_DAY = "20260105"


class FrozenDatetime(real_datetime.datetime):
    """`datetime.datetime` whose `now()` is fixed at 2026-01-05 12:00."""

    @classmethod
    def now(cls, tz: Optional[real_datetime.tzinfo] = None) -> "FrozenDatetime":
        """
        Return the frozen moment.

        :param tz: Optional tzinfo, attached as given.
        :return: 2026-01-05 12:00:00.
        """
        return cls(2026, 1, 5, 12, 0, 0, tzinfo=tz)


def bucket_keys(client: object) -> List[str]:
    """
    Every key in `BUCKET`, sorted.

    :param client: An S3 client on the fake gateway.
    :return: The keys.
    """
    listing = client.list_objects_v2(Bucket=BUCKET)
    return sorted(obj["Key"] for obj in listing.get("Contents", []))


class FlakyStorj:
    """
    The moto client, with chosen calls failing the way a Storj outage does.

    :param s3: The moto client every other call goes to.
    :param read: A key whose `get_object` raises.
    :param write: A key whose `upload_fileobj` raises.
    :param copy: Whether `copy_object` (the backup) raises.
    """

    def __init__(
        self,
        s3: object,
        read: Optional[str] = None,
        write: Optional[str] = None,
        copy: bool = False,
    ) -> None:
        self.s3 = s3
        self.read = read
        self.write = write
        self.copy = copy

    def __getattr__(self, name: str) -> object:
        return getattr(self.s3, name)

    def get_object(self, **kwargs: object) -> Dict:
        if kwargs["Key"] == self.read:
            raise OSError("storj read timed out")
        return self.s3.get_object(**kwargs)

    def upload_fileobj(self, **kwargs: object) -> None:
        if kwargs["Key"] == self.write:
            raise OSError("storj write refused")
        return self.s3.upload_fileobj(**kwargs)

    def copy_object(self, **kwargs: object) -> Dict:
        if self.copy:
            raise OSError("storj copy refused")
        return self.s3.copy_object(**kwargs)


def flaky_storj(monkeypatch: pytest.MonkeyPatch, s3: object, **fail: object) -> None:
    """
    Route every `storj` client through a `FlakyStorj`.

    :param monkeypatch: The test's monkeypatch.
    :param s3: The moto client.
    :param fail: `FlakyStorj` keyword arguments.
    """
    flaky = FlakyStorj(s3, **fail)
    monkeypatch.setattr(storj, "get_s3_client", lambda: flaky)
