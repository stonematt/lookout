# Updated Coding Standards for Lookout - Weather Dashboard Development

## Introduction

This document outlines the revised coding standards for the Lookout project, focusing on Python development for weather dashboards with an emphasis on Pandas, NumPy, Streamlit.io, Storj.io for distributed storage, and Ambient Weather Network via AmbientAPI.

## Code Style and Quality

- **Flake8 Compliance**: All code must follow flake8 standards.
- **Black Formatter**: Use Black for consistent code formatting.

## Python Libraries

- **Pandas and NumPy**: For data manipulation and numerical analysis.
- **Streamlit Integration**: For creating interactive dashboards.

## Data Handling and Storage

- **AmbientAPI**: For fetching weather data.
- **Storj.io Storage**: Use Storj.io as a decentralized, S3-compatible storage system.

## Code Structure and Organization

- **Modularity**: Ensure a modular and well-organized codebase.
- **Documentation**: Provide clear, concise comments and docstrings. Keep line lengths under 88 characters.

## Module Documentation

- **Purpose and Functionality**: Begin each module with a docstring detailing its purpose and key functionalities.
- **Integration Points**: Highlight integration with other system parts (e.g., Streamlit, S3 storage).
- **List of Functions**: Concisely list and describe each function or class.
- **Example Usage**: Include examples or references for using the module or key functions.

### Example Format

```python
"""
storj_df_s3.py: This module interacts with S3 storage, focusing on dataframe (pandas) and JSON object handling. It provides functionalities for reading and writing to S3 buckets, using s3fs and pandas. It integrates with Streamlit for secure access to S3. The module is essential for tasks like data backup, retrieval, and storage.

Functions:
- Manage JSON files with S3.
- Handle pandas DataFrames for S3 storage.
- Enumerate S3 bucket items.
- Backup S3 data.
"""
```

## Error Handling and Logging

- **Comprehensive Error Handling**: Implement thorough error handling.
- **Logging**: Use logging information for debugging rather than print statements.

## Environment and Tools

- **Conda Environment**: Manage dependencies with Conda.
- **VS Code on Mac**: Develop with Visual Studio Code on macOS.

## Docstring and Commenting

- **Docstring Format**: Use concise docstrings with type hinting for function declarations.

    ```python
    def function_name(param1: type, param2: type) -> return_type:
        """
        Briefly describe the function.

        :param param1: Description of param1.
        :param param2: Description of param2.
        :return: Description of the return value.
        """
        # Inline comments for context
        # Function implementation
    ```

- **Inline Comments**: Include inline comments to maintain context and process understanding. Keep them concise and within the 88-character limit.

These updated standards aim to enhance the readability, maintainability, and efficiency of the Lookout project's codebase.
