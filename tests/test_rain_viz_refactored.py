"""
Unit tests for refactored rain_viz functions.

Tests that the refactored functions still work correctly and produce
the same output as before refactoring.
"""

import pytest
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timezone

from lookout.core.rain_viz import create_event_accumulation_chart


class TestRefactoredFunctions:
    """Test refactored visualization functions."""
    
    def setup_method(self):
        """Set up test data for each test method."""
        # Create sample event data
        base_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        timestamps_ms = [
            int(base_time.timestamp() * 1000) + i * 300000  # 5-minute intervals
            for i in range(10)
        ]
        
        self.sample_event_data = pd.DataFrame({
            'dateutc': timestamps_ms,
            'eventrainin': [0.0, 0.05, 0.12, 0.18, 0.25, 0.31, 0.35, 0.38, 0.40, 0.42],
            'dailyrainin': [0.0, 0.05, 0.07, 0.06, 0.07, 0.06, 0.04, 0.03, 0.02, 0.02]
        })
        
        self.sample_event_info = {
            'total_rainfall': 0.42,
            'duration_minutes': 45,
            'start_time': base_time,
            'end_time': datetime(2024, 1, 15, 10, 45, 0, tzinfo=timezone.utc)
        }
    
    def test_create_event_accumulation_chart_basic(self):
        """Test basic functionality of refactored create_event_accumulation_chart."""
        fig = create_event_accumulation_chart(self.sample_event_data, self.sample_event_info)
        
        # Verify it returns a Plotly figure
        assert isinstance(fig, go.Figure)
        
        # Verify it has the expected trace
        assert len(fig.data) == 1
        assert fig.data[0].type == "scatter"
        assert fig.data[0].mode == "lines"
        assert fig.data[0].fill == "tozeroy"
        
        # Verify layout properties
        assert fig.layout.height == 300
        assert fig.layout.showlegend is False
        assert fig.layout.hovermode == "x unified"
        
        # Verify axis properties
        assert fig.layout.xaxis.title.text == ""
        assert fig.layout.yaxis.title.text == "Rainfall (in)"
        assert fig.layout.xaxis.showgrid is False
        assert fig.layout.yaxis.showgrid is True
        assert fig.layout.yaxis.rangemode == "tozero"
        
        # Verify annotation is present
        assert len(fig.layout.annotations) == 1
        assert "Total: 0.420\"" in fig.layout.annotations[0].text
    
    def test_create_event_accumulation_chart_colors(self):
        """Test that refactored function uses standard colors."""
        fig = create_event_accumulation_chart(self.sample_event_data, self.sample_event_info)
        
        # Get the trace
        trace = fig.data[0]
        
        # Verify colors are from standard palette (not hardcoded)
        assert trace.line.color is not None
        assert trace.fillcolor is not None
        
        # These should match the standard colors from chart_config
        from lookout.core.chart_config import get_standard_colors
        colors = get_standard_colors()
        
        assert trace.line.color == colors["rainfall_line"]
        assert trace.fillcolor == colors["rainfall_fill"]
    
    def test_create_event_accumulation_chart_annotation_position(self):
        """Test that annotation uses standard positioning."""
        fig = create_event_accumulation_chart(self.sample_event_data, self.sample_event_info)
        
        annotation = fig.layout.annotations[0]
        
        # Verify annotation uses standard positioning from chart_config
        assert annotation.xref == "paper"
        assert annotation.yref == "paper"
        assert annotation.x == 0.98
        assert annotation.y == 0.95
        assert annotation.xanchor == "right"
        assert annotation.yanchor == "top"
        assert annotation.showarrow is False
    
    def test_create_event_accumulation_chart_empty_data(self):
        """Test function with empty event data."""
        empty_data = pd.DataFrame({
            'dateutc': [],
            'eventrainin': [],
            'dailyrainin': []
        })
        empty_info = {'total_rainfall': 0.0, 'duration_minutes': 0}
        
        fig = create_event_accumulation_chart(empty_data, empty_info)
        
        # Should still return a valid figure
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1  # Still creates the trace
        assert len(fig.layout.annotations) == 1  # Still adds annotation
    
    def test_create_event_accumulation_chart_single_point(self):
        """Test function with single data point."""
        single_point_data = pd.DataFrame({
            'dateutc': [int(datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)],
            'eventrainin': [0.1],
            'dailyrainin': [0.1]
        })
        single_info = {'total_rainfall': 0.1, 'duration_minutes': 5}
        
        fig = create_event_accumulation_chart(single_point_data, single_info)
        
        # Should still return a valid figure
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert len(fig.layout.annotations) == 1
        assert "Total: 0.100\"" in fig.layout.annotations[0].text


class TestBackwardCompatibility:
    """Test that refactored functions maintain backward compatibility."""
    
    def test_function_signature_unchanged(self):
        """Test that function signature hasn't changed."""
        import inspect
        
        sig = inspect.signature(create_event_accumulation_chart)
        params = list(sig.parameters.keys())
        
        # Should have the same parameters as before
        assert params == ['event_data', 'event_info']
        
        # Should have the same return type annotation
        assert sig.return_annotation == go.Figure
    
    def test_output_structure_unchanged(self):
        """Test that output structure is the same as before."""
        # Create test data
        base_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        test_data = pd.DataFrame({
            'dateutc': [int(base_time.timestamp() * 1000)],
            'eventrainin': [0.1],
            'dailyrainin': [0.1]
        })
        test_info = {'total_rainfall': 0.1, 'duration_minutes': 5}
        
        fig = create_event_accumulation_chart(test_data, test_info)
        
        # Verify the structure is what other code expects
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')
        assert hasattr(fig.layout, 'annotations')
        assert hasattr(fig.layout, 'xaxis')
        assert hasattr(fig.layout, 'yaxis')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])