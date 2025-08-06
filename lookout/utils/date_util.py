from dateutil import parser

from lookout.utils.log_util import app_logger

logger = app_logger(__name__)


def to_date(date_string: str):
    """
    Convert a date string to a datetime object.

    :param date_string: str - The date string to parse.
    :return: datetime - Parsed datetime object.
    :raises: Exception if date string parsing fails.
    """
    try:
        return parser.parse(date_string)
    except Exception as e:
        logger.error(f"Error parsing date string: {e}", exc_info=True)
        raise
