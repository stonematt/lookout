"""
storage.py: Helpers for tests that run against the `fake_s3` fixture.

`BUCKET` is the bucket `fake_s3` creates. `bucket_keys(client)` lists it.
`FrozenDatetime` stands in for `datetime.datetime` with a fixed `now()`; the
`frozen_today` fixture in conftest installs it where `backup_data` reads it.
"""

import datetime as real_datetime
from typing import List, Optional

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
