# Coding Standards for Lookout - Weather Dashboard Development

## Introduction

This document establishes coding standards for the Lookout project, tailored for developers and designers working on weather dashboards. It focuses on using Python with Pandas, NumPy, Streamlit.io, Storj.io for distributed storage, and the Ambient Weather Network via the AmbientAPI.

## Code Style and Quality

- **Flake8 Compliance**: All code must adhere to flake8 standards.
- **Black Formatter**: Use Black for uniform code formatting.

## Python Libraries

- **Pandas and NumPy**: For data manipulation and numerical calculations.
- **Streamlit Integration**: For building interactive dashboards.

## Data Handling and Storage

- **AmbientAPI**: For retrieving weather data.
- **Storj.io Storage**: Utilize Storj.io as a decentralized, S3-compatible storage solution.

## Code Structure and Organization

- **Modularity**: Maintain a modular and organized codebase.
- **Documentation**: Clear, concise comments and docstrings for all code segments.

## Error Handling and Logging

- **Comprehensive Error Handling**: Implement robust error handling.
- **Logging**: Effective use of logging for debugging and tracking.

## Environment and Tools

- **Conda Environment**: Use Conda for managing dependencies.
- **VS Code on Mac**: Develop using Visual Studio Code on macOS.

## Docstring Format

Use the following format for docstrings:

```python
def function_name(param1, param2):
    """
    A brief description of the function.

    :param param1: type - Description of param1.
    :param param2: type - Description of param2.
    :return: return_type - Description of the return value.
    """
    # Function implementation
```

### Example

```python
def fetch_weather_data(api_key, station_id):
    """
    Retrieve weather data from Ambient Weather Network using API key and station ID.

    :param api_key: str - The API key for authentication.
    :param station_id: str - The unique identifier for the weather station.
    :return: dict - Weather data retrieved from the specified station.
    """
    # Function implementation...
```
