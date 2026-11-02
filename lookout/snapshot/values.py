"""
Small value helpers every snapshot section shares: one rounding rule and one
ISO-date reader, so a missing number is `None` the same way on every tab.
"""

from datetime import date
from typing import Any, Optional

import numpy as np


def number(value: Any, digits: int = 3) -> Optional[float]:
    """
    A float rounded to `digits`, or None for a missing, NaN or unreadable value.

    :param value: A number, or anything `float` reads, or None.
    :param digits: Decimal places to keep.
    :return: The rounded float, or None.
    """
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(result) else round(result, digits)


def iso_day(iso: str) -> date:
    """A date from its ISO form."""
    return date.fromisoformat(iso)
