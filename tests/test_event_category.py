"""
A rain event's category from its total and peak rate: sprinkle, shower,
rain or storm. A sprinkle is judged on its total alone; above it, the larger
of the total's and the peak rate's category wins.
"""

import pytest

from lookout.core.event_category import CATEGORIES, event_category


@pytest.mark.parametrize(
    "total_in, peak_rate_in_hr, expected",
    [
        (0.01, 0.0, "sprinkle"),
        (0.04, 0.24, "sprinkle"),
        (0.04, 1.5, "sprinkle"),
        (0.05, 0.12, "shower"),
        (0.49, 0.49, "shower"),
        (0.12, None, "shower"),
        (0.5, 0.1, "rain"),
        (0.2, 0.5, "rain"),
        (1.99, 0.99, "rain"),
        (2.0, 0.2, "storm"),
        (0.37, 1.08, "storm"),
        (5.16, 0.61, "storm"),
    ],
)
def test_an_event_takes_the_category_of_its_total_or_its_peak_rate(
    total_in: float, peak_rate_in_hr: float, expected: str
) -> None:
    assert event_category(total_in, peak_rate_in_hr) == expected


def test_the_categories_run_smallest_first() -> None:
    assert CATEGORIES == ("sprinkle", "shower", "rain", "storm")
