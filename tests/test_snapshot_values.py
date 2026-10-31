"""The shared value helpers every snapshot section rounds and reads days with."""

from datetime import date

import numpy as np

from lookout.snapshot.values import iso_day, number


def test_number_rounds_and_keeps_missing_as_none():
    assert number(1.23456) == 1.235
    assert number(1.23456, 1) == 1.2
    assert number("2.5") == 2.5
    assert number(None) is None
    assert number(np.nan) is None
    assert number("n/a") is None


def test_iso_day_reads_an_iso_date():
    assert iso_day("2026-01-05") == date(2026, 1, 5)
