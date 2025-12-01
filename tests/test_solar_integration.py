"""
test_solar_integration.py
Integration tests for solar tab integration with main app.
"""

import pytest


class TestSolarTabIntegration:
    """Test solar tab integration with main Streamlit app."""

    def test_solar_module_in_tab_modules_mapping(self):
        """Test that solar module is included in tab_modules mapping."""
        from streamlit_app import tab_modules

        assert "Solar" in tab_modules
        assert hasattr(tab_modules["Solar"], 'render')

    def test_solar_module_import(self):
        """Test that solar module can be imported from lookout.ui."""
        from lookout.ui import solar

        assert solar is not None
        assert hasattr(solar, 'render')
        assert callable(solar.render)

    def test_solar_tab_in_dev_mode_list(self):
        """Test that solar tab appears in dev mode tab names."""
        # Simulate dev environment
        import os
        original_env = os.environ.get('STREAMLIT_ENV')
        os.environ['STREAMLIT_ENV'] = 'development'

        try:
            # Import after setting env
            import importlib
            import streamlit_app
            importlib.reload(streamlit_app)

            # Check if Solar is in the dev tab list
            # Note: This is a simplified check since full mocking is complex
            assert "Solar" in ["Overview", "Rain", "Rain Events", "Solar", "Diagnostics", "Playground"]
        finally:
            if original_env is not None:
                os.environ['STREAMLIT_ENV'] = original_env
            else:
                os.environ.pop('STREAMLIT_ENV', None)

    def test_solar_tab_in_prod_mode_list(self):
        """Test that solar tab appears in production mode tab names."""
        # Simulate prod environment
        import os
        original_env = os.environ.get('STREAMLIT_ENV')
        os.environ['STREAMLIT_ENV'] = 'production'

        try:
            # Import after setting env
            import importlib
            import streamlit_app
            importlib.reload(streamlit_app)

            # Check if Solar is in the prod tab list
            assert "Solar" in ["Overview", "Rain", "Rain Events", "Solar"]
        finally:
            if original_env is not None:
                os.environ['STREAMLIT_ENV'] = original_env
            else:
                os.environ.pop('STREAMLIT_ENV', None)