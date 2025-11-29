"""
Unit tests for chart configuration helpers.

Tests all helper functions in lookout.core.chart_config to ensure
they return valid Plotly configuration objects.
"""

import pytest
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lookout.core.chart_config import (
    get_default_margins,
    get_standard_colors,
    apply_time_series_layout,
    apply_standard_axes,
    create_standard_annotation,
    apply_heatmap_layout,
    get_rainfall_colorscale,
    apply_violin_layout,
)


class TestDefaultMargins:
    """Test get_default_margins helper function."""
    
    def test_default_margins(self):
        """Test default margin configuration."""
        margins = get_default_margins()
        
        assert isinstance(margins, dict)
        assert "l" in margins  # left
        assert "r" in margins  # right
        assert "t" in margins  # top
        assert "b" in margins  # bottom
        assert all(isinstance(v, (int, float)) for v in margins.values())
        assert margins == {"l": 50, "r": 20, "t": 40, "b": 40}
    
    def test_compact_margins(self):
        """Test compact margin configuration."""
        margins = get_default_margins(compact=True)
        
        assert isinstance(margins, dict)
        assert margins == {"l": 30, "r": 20, "t": 30, "b": 40}


class TestStandardColors:
    """Test get_standard_colors helper function."""
    
    def test_default_colors(self):
        """Test default color scheme."""
        colors = get_standard_colors()
        
        assert isinstance(colors, dict)
        assert "muted_line" in colors
        assert "rainfall_line" in colors
        assert "rainfall_fill" in colors
        assert "rate_low" in colors
        assert "rate_medium" in colors
        assert "rate_high" in colors
        assert len(colors) >= 8
    
    def test_color_values(self):
        """Test that color values are valid color codes."""
        colors = get_standard_colors()
        
        for color_name, color_value in colors.items():
            assert isinstance(color_value, str)
            # Check for hex codes or rgba values
            assert color_value.startswith("#") or color_value.startswith("rgba")


class TestTimeSeriesLayout:
    """Test apply_time_series_layout helper function."""
    
    def test_default_layout(self):
        """Test layout with default parameters."""
        fig = go.Figure()
        result_fig = apply_time_series_layout(fig)
        
        assert isinstance(result_fig, go.Figure)
        assert result_fig.layout.height == 450
        assert result_fig.layout.showlegend is False
        assert result_fig.layout.hovermode == "x unified"
        # Template is set (either as string or Template object)
        assert hasattr(result_fig.layout, 'template')
    
    def test_custom_parameters(self):
        """Test layout with custom parameters."""
        fig = go.Figure()
        custom_title = "Custom Chart Title"
        result_fig = apply_time_series_layout(
            fig, 
            height=500, 
            showlegend=True, 
            title=custom_title,
            compact=True
        )
        
        assert result_fig.layout.height == 500
        assert result_fig.layout.showlegend is True
        assert result_fig.layout.title.text == custom_title
        # Check margins are set (as Margin object)
        assert hasattr(result_fig.layout, 'margin')


class TestStandardAxes:
    """Test apply_standard_axes helper function."""
    
    def test_default_axes(self):
        """Test default axis configuration."""
        fig = go.Figure()
        result_fig = apply_standard_axes(fig)
        
        assert isinstance(result_fig, go.Figure)
        assert result_fig.layout.xaxis.title.text == ""
        assert result_fig.layout.yaxis.title.text == ""
        assert result_fig.layout.xaxis.showgrid is False
        assert result_fig.layout.yaxis.showgrid is True
        assert result_fig.layout.yaxis.rangemode == "tozero"
    
    def test_custom_axes(self):
        """Test custom axis configuration."""
        fig = go.Figure()
        result_fig = apply_standard_axes(
            fig,
            xaxis_title="Time",
            yaxis_title="Rainfall (in)",
            showgrid_x=True,
            showgrid_y=False,
            type_x="date"
        )
        
        assert result_fig.layout.xaxis.title.text == "Time"
        assert result_fig.layout.yaxis.title.text == "Rainfall (in)"
        assert result_fig.layout.xaxis.showgrid is True
        assert result_fig.layout.yaxis.showgrid is False
        assert result_fig.layout.xaxis.type == "date"


class TestStandardAnnotation:
    """Test create_standard_annotation helper function."""
    
    def test_default_annotation(self):
        """Test default annotation configuration."""
        annotation = create_standard_annotation("Test Text")
        
        assert isinstance(annotation, dict)
        assert annotation["text"] == "Test Text"
        assert annotation["xref"] == "paper"
        assert annotation["yref"] == "paper"
        assert annotation["showarrow"] is False
        assert annotation["x"] == 0.98
        assert annotation["y"] == 0.95
        assert annotation["xanchor"] == "right"
        assert annotation["yanchor"] == "top"
    
    def test_position_presets(self):
        """Test different position presets."""
        # Test top_left
        annotation_tl = create_standard_annotation("Test", position="top_left")
        assert annotation_tl["x"] == 0.02
        assert annotation_tl["xanchor"] == "left"
        
        # Test bottom_right
        annotation_br = create_standard_annotation("Test", position="bottom_right")
        assert annotation_br["y"] == 0.05
        assert annotation_br["yanchor"] == "bottom"
    
    def test_custom_parameters(self):
        """Test annotation with custom parameters."""
        annotation = create_standard_annotation(
            "Test",
            showarrow=True,
            bgcolor="yellow",
            custom_param="value"
        )
        
        assert annotation["showarrow"] is True
        assert annotation["bgcolor"] == "yellow"
        assert annotation["custom_param"] == "value"


class TestHeatmapLayout:
    """Test apply_heatmap_layout helper function."""
    
    def test_default_heatmap(self):
        """Test default heatmap configuration."""
        fig = go.Figure()
        result_fig = apply_heatmap_layout(
            fig,
            title="Heatmap Test",
            xaxis_title="Hour",
            yaxis_title="Day"
        )
        
        assert isinstance(result_fig, go.Figure)
        assert result_fig.layout.title.text == "Heatmap Test"
        assert result_fig.layout.height == 600
        assert result_fig.layout.xaxis.title.text == "Hour"
        assert result_fig.layout.yaxis.title.text == "Day"
        assert result_fig.layout.xaxis.type == "category"
        assert result_fig.layout.yaxis.type == "category"
        assert result_fig.layout.yaxis.autorange == "reversed"
    
    def test_custom_heatmap(self):
        """Test heatmap with custom parameters."""
        fig = go.Figure()
        result_fig = apply_heatmap_layout(
            fig,
            title="Custom Heatmap",
            xaxis_title="X",
            yaxis_title="Y",
            height=400,
            compact=True,
            showlegend=False,
            colorbar_title="Custom Units"
        )
        
        assert result_fig.layout.height == 400
        assert result_fig.layout.showlegend is False
        # Check margins are set
        assert hasattr(result_fig.layout, 'margin')
        assert result_fig.layout.coloraxis.colorbar.title.text == "Custom Units"


class TestRainfallColorscale:
    """Test get_rainfall_colorscale helper function."""
    
    def test_colorscale_structure(self):
        """Test colorscale structure."""
        colorscale = get_rainfall_colorscale()
        
        assert isinstance(colorscale, list)
        assert len(colorscale) >= 2
        
        # Each item should be a 2-element list [value, color]
        for item in colorscale:
            assert isinstance(item, list)
            assert len(item) == 2
            assert isinstance(item[0], (int, float))  # value
            assert isinstance(item[1], str)           # color
            assert 0 <= item[0] <= 1                 # value range
    
    def test_colorscale_range(self):
        """Test colorscale value range."""
        colorscale = get_rainfall_colorscale()
        
        values = [item[0] for item in colorscale]
        assert min(values) == 0.0
        assert max(values) == 1.0


class TestViolinLayout:
    """Test apply_violin_layout helper function."""
    
    def test_default_violin(self):
        """Test default violin layout configuration."""
        fig = go.Figure()
        result_fig = apply_violin_layout(fig)
        
        assert isinstance(result_fig, go.Figure)
        assert result_fig.layout.height == 500
        assert result_fig.layout.yaxis.title.text == ""
        assert result_fig.layout.xaxis.title.text == ""
        assert result_fig.layout.showlegend is True
        # Check template is set
        assert hasattr(result_fig.layout, 'template')
        assert result_fig.layout.hovermode == "x unified"
    
    def test_custom_violin(self):
        """Test violin layout with custom parameters."""
        fig = go.Figure()
        custom_title = "Violin Plot Test"
        result_fig = apply_violin_layout(
            fig,
            title=custom_title,
            height=600,
            yaxis_title="Values",
            showlegend=False
        )
        
        assert result_fig.layout.height == 600
        assert result_fig.layout.title.text == custom_title
        assert result_fig.layout.yaxis.title.text == "Values"
        assert result_fig.layout.showlegend is False


class TestIntegration:
    """Integration tests for combined helper usage."""
    
    def test_complete_chart_setup(self):
        """Test creating a complete chart setup using multiple helpers."""
        # Create figure
        fig = go.Figure()
        
        # Apply time series layout
        fig = apply_time_series_layout(
            fig,
            title="Integration Test Chart",
            height=400,
            showlegend=True
        )
        
        # Apply standard axes
        fig = apply_standard_axes(
            fig,
            xaxis_title="Time",
            yaxis_title="Rainfall (in)"
        )
        
        # Add annotation
        annotation = create_standard_annotation(
            "Test Annotation",
            position="top_left"
        )
        fig.add_annotation(annotation)
        
        # Verify final figure
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Integration Test Chart"
        assert fig.layout.height == 400
        assert fig.layout.showlegend is True
        assert fig.layout.xaxis.title.text == "Time"
        assert fig.layout.yaxis.title.text == "Rainfall (in)"
        assert len(fig.layout.annotations) == 1
    
    def test_subplot_with_multiple_configs(self):
        """Test creating subplots with multiple configurations."""
        fig = make_subplots(rows=2, cols=1)
        
        # Apply time series layout to subplots
        fig = apply_time_series_layout(fig, title="Subplot Test")
        
        # Apply axes to both subplots
        fig = apply_standard_axes(
            fig,
            xaxis_title="Time",
            yaxis_title="Rainfall",
            type_x="date"
        )
        
        # Verify configuration
        assert fig.layout.title.text == "Subplot Test"
        assert fig.layout.xaxis.type == "date"
        assert fig.layout.xaxis.title.text == "Time"
        assert fig.layout.yaxis.title.text == "Rainfall"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])