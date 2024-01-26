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

## Module Documentation

- **Purpose and Functionality**: Each module must begin with a docstring explaining its purpose and main functionalities.
- **Integration Points**: Mention how the module integrates with other parts of the system (e.g., Streamlit, S3 storage).
- **List of Functions**: Briefly list and describe each function or class within the module.
- **Example Usage**: Provide a simple example or reference to demonstrate how the module or its key functions are used in the broader application.

### Example Format

```python
"""
storj_df_s3.py: this module provides functionalities to interact with s3 storage,
specifically for handling dataframes (pandas) and json objects.  It includes utilities
for reading and writing data to/from s3 buckets using s3fs and pandas.  It also
features logging setup for monitoring file operations and a backup mechanism for s3
data.  The module integrates with streamlit for secret management, enhancing the
security aspect of s3 access.  Designed for applications where s3 storage interaction
is crucial, this module streamlines tasks like data backup, retrieval, and storage
of structured data.

Functions:
- Read and write JSON files to/from S3.
- Convert pandas DataFrames to/from JSON or Parquet format for S3 storage.
- List items in an S3 bucket.
- Backup data from S3 to a designated backup directory.
"""
```

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
