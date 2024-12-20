# config.py
"""
Configurations for the Lookout weather dashboard application.

This module contains the sensor index and other configurations that are shared
across the application, such as Streamlit visualization settings and data
processing parameters.
"""

sensor_index = {
    "location1": {
        "name": "Outdoor",
        "sensors": {
            "temperature": {
                "tempf": "Temperature",
                "feelsLike": "Feels Like",
                "dewPoint": "Dew Point",
            },
            "humidity": {
                "humidity": "Humidity",
            },
            "wind": {
                "winddir": "Wind Direction",
                "windspeedmph": "Wind Speed",
                "windgustmph": "Wind Gust",
                "maxdailygust": "Max Daily Gust",
            },
            "rain": {
                "hourlyrainin": "Hourly Rain",
                "eventrainin": "Event Rain",
                "dailyrainin": "Daily Rain",
                "weeklyrainin": "Weekly Rain",
                "monthlyrainin": "Monthly Rain",
                "yearlyrainin": "Yearly Rain",
            },
            "pressure": {
                "baromrelin": "Relative Pressure",
                "baromabsin": "Absolute Pressure",
            },
            "solar": {
                "solarradiation": "Solar Radiation",
            },
            "uv": {
                "uv": "UV Index",
            }
        }
    },
    "location2": {
        "name": "Indoor",
        "sensors": {
            "temperature": {
                "tempinf": "Indoor Temperature",
                "feelsLikein": "Indoor Feels Like",
                "dewPointin": "Indoor Dew Point",
            },
            "humidity": {
                "humidityin": "Indoor Humidity",
            },
            "battery": {
                "battin": "Battery Status",
            },
        }
    },
    "location3": {
        "name": "Office",
        "sensors": {
            "temperature": {
                "temp1f": "Office Temperature",
                "feelsLike1": "Office Feels Like",
                "dewPoint1": "Office Dew Point",
            },
            "humidity": {
                "humidity1": "Office Humidity",
            },
            "battery": {
                "batt1": "Office Battery Status",
            },
        }
    },
    "timestamps": {
        "last_data_collection": {
            "dateutc": "UTC Date Time",
            "date": "Local Date Time",
            "lastRain": "Last Rain Event",
        }
    }
}
