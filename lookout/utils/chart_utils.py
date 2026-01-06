"""
Chart utilities for visualizations.

Provides utility functions for chart configuration, axis calculations,
and visualization helpers used across different modules.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from lookout.utils.log_util import app_logger

logger = app_logger(__name__)


def get_global_axis_range(all_periods_data: List[List[float]]) -> Tuple[float, float]:
    """
    Calculate consistent y-axis range across all sparklines for intuitive comparison.

    :param all_periods_data: List of lists containing values for each period [7d, 30d, 365d]
    :return: Tuple of (min_value, max_value) for fixed axis
    """
    # Flatten all values and filter out NaN
    all_values = []
    for period_data in all_periods_data:
        all_values.extend([v for v in period_data if not pd.isna(v)])

    if not all_values:
        return (0, 1.0)  # Default range if no data

    # Find max value
    max_value = max(all_values)

    # Round up to nice number for axis
    if max_value <= 0.5:
        axis_max = 0.5
    elif max_value <= 1.0:
        axis_max = 1.0
    elif max_value <= 2.0:
        axis_max = 2.0
    elif max_value <= 5.0:
        axis_max = 5.0
    elif max_value <= 10.0:
        axis_max = 10.0
    elif max_value <= 20.0:
        axis_max = 20.0
    elif max_value <= 50.0:
        axis_max = 50.0
    elif max_value <= 100.0:
        axis_max = 100.0
    else:
        # Round up to nearest 100
        axis_max = np.ceil(max_value / 100) * 100

    return (0, axis_max)