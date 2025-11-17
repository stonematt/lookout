"""
Centralized style management for weather dashboard components.

This module provides a scalable CSS architecture following SOLID principles
for consistent styling across the application.
"""

import streamlit as st
from dataclasses import dataclass
from typing import Dict, Optional
import re

from lookout.utils.log_util import app_logger

logger = app_logger(__name__)


@dataclass
class StyleConfig:
    """Configuration dataclass for style parameters."""
    
    # Weather header styles
    header_font_size: str = "0.9rem"
    header_line_height: str = "1.2"
    header_letter_spacing: str = "0.5px"
    metric_spacing: str = "0.5rem"
    separator_margin: str = "0.3rem"
    
    # Colors
    text_color: str = "#262730"
    separator_color: str = "#666666"
    active_event_bg: str = "#fff3cd"
    active_event_border: str = "#ffeaa7"
    
    # Responsive breakpoints
    mobile_breakpoint: str = "768px"
    tablet_breakpoint: str = "1024px"


class StyleManager:
    """
    Centralized style management with singleton pattern.
    
    Provides CSS injection and HTML rendering utilities for consistent
    component styling across application. Follows SOLID principles:
    
    - Single Responsibility: Manages only CSS/HTML rendering
    - Open/Closed: Extensible without modifying existing code
    - Dependency Inversion: Components depend on abstraction, not concrete CSS
    
    Usage:
        style_manager = get_style_manager()
        style_manager.inject_styles()  # Call once per session
        style_manager.render_weather_header(html_content)
    
    CSS Classes:
        - .weather-header: Main container for weather display
        - .active-event-banner: Styled banner for active rain events
        - .current-conditions: Container for current weather metrics
        - .weather-metrics-line: Flex container for metric groups
        - .metric-group: Individual metric with emoji and value
        - .metric-separator: Visual separator between metrics
    
    Responsive Breakpoints:
        - Desktop: >768px (horizontal layout, normal fonts)
        - Tablet: ≤768px (tighter spacing, smaller fonts)
        - Mobile: ≤480px (vertical layout, hidden separators)
        - Tiny: ≤320px (ultra-compact layout)
    """
    
    _instance: Optional['StyleManager'] = None
    _initialized: bool = False
    
    def __new__(cls) -> 'StyleManager':
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize StyleManager with default configuration."""
        if not hasattr(self, '_config'):
            self._config = StyleConfig()
    
    @property
    def config(self) -> StyleConfig:
        """Get current style configuration."""
        return self._config
    
    def inject_styles(self) -> None:
        """Inject global CSS styles once per session."""
        if "styles_injected" in st.session_state:
            return
            
        css = self._generate_css()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
        st.session_state["styles_injected"] = True
        logger.debug("Global CSS styles injected")
    
    def _generate_css(self) -> str:
        """Generate CSS rules from configuration."""
        return f"""
        /* Weather Header Styles */
        .weather-header {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            font-size: {self.config.header_font_size};
            line-height: {self.config.header_line_height};
            letter-spacing: {self.config.header_letter_spacing};
            color: {self.config.text_color};
            margin: 0.5rem 0;
        }}
        
        .metric-group {{
            display: inline-block;
            white-space: nowrap;
            margin-right: {self.config.metric_spacing};
        }}
        
        .metric-separator {{
            color: {self.config.separator_color};
            margin: 0 {self.config.separator_margin};
            font-weight: 500;
        }}
        
        .current-conditions {{
            margin: 0.25rem 0;
        }}
        
        .active-event-banner {{
            background-color: {self.config.active_event_bg};
            border: 1px solid {self.config.active_event_border};
            border-radius: 6px;
            padding: 0.5rem 0.75rem;
            margin: 0.5rem 0;
            font-weight: 500;
            font-size: 0.85rem;
            line-height: 1.3;
        }}
        
        .weather-metrics-line {{
            margin: 0.25rem 0;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: {self.config.metric_spacing};
        }}
        
        /* Responsive Design */
        @media (max-width: {self.config.mobile_breakpoint}) {{
            .weather-header {{
                font-size: 0.8rem;
                line-height: 1.1;
                margin: 0.3rem 0;
            }}
            
            .active-event-banner {{
                font-size: 0.75rem;
                padding: 0.4rem 0.6rem;
                line-height: 1.2;
                margin: 0.3rem 0;
            }}
            
            .weather-metrics-line {{
                gap: 0.25rem;
                margin: 0.2rem 0;
            }}
            
            .metric-group {{
                margin-right: 0.25rem;
            }}
            
            .metric-separator {{
                margin: 0 0.2rem;
            }}
        }}
        
        @media (max-width: 480px) {{
            .weather-header {{
                font-size: 0.75rem;
                line-height: 1.0;
                margin: 0.2rem 0;
            }}
            
            .active-event-banner {{
                font-size: 0.7rem;
                padding: 0.3rem 0.5rem;
                margin: 0.2rem 0;
            }}
            
            .weather-metrics-line {{
                gap: 0.2rem;
                flex-direction: column;
                align-items: flex-start;
            }}
            
            .metric-group {{
                margin-right: 0;
                margin-bottom: 0.15rem;
            }}
            
            .metric-separator {{
                display: none; /* Hide separators on small mobile */
            }}
        }}
        
        @media (max-width: 320px) {{
            .weather-header {{
                font-size: 0.7rem;
            }}
            
            .active-event-banner {{
                font-size: 0.65rem;
                padding: 0.25rem 0.4rem;
            }}
        }}
        """
    
    def render_weather_header(self, html_content: str) -> None:
        """
        Render weather header with proper CSS classes.
        
        :param html_content: HTML content for weather header
        """
        wrapped_html = f'<div class="weather-header">{html_content}</div>'
        st.markdown(wrapped_html, unsafe_allow_html=True)
    
    def render_active_event_banner(self, html_content: str) -> None:
        """
        Render active event banner with styling.
        
        :param html_content: HTML content for active event
        """
        wrapped_html = f'<div class="active-event-banner">{html_content}</div>'
        st.markdown(wrapped_html, unsafe_allow_html=True)
    
    def render_current_conditions(self, html_content: str) -> None:
        """
        Render current conditions with proper styling.
        
        :param html_content: HTML content for current conditions
        """
        wrapped_html = f'<div class="current-conditions">{html_content}</div>'
        st.markdown(wrapped_html, unsafe_allow_html=True)
    
    def build_metric_group(self, emoji: str, value: str, unit: str = "", trend: str = "") -> str:
        """
        Build HTML for a metric group with emoji and value.
        
        :param emoji: Emoji icon for the metric
        :param value: Metric value
        :param unit: Optional unit string
        :param trend: Optional trend indicator
        :return: HTML string for the metric group
        """
        parts = [emoji]
        if value:
            parts.append(value)
        if unit:
            parts.append(unit)
        if trend:
            parts.append(trend)
        
        content = "\u00a0".join(parts)  # Non-breaking spaces
        return f'<span class="metric-group">{content}</span>'
    
    def build_separator(self) -> str:
        """
        Build HTML for a separator between metrics.
        
        :return: HTML string for separator
        """
        return f'<span class="metric-separator">\u00a0•\u00a0</span>'
    
    def build_metrics_line(self, metric_groups: list[str]) -> str:
        """
        Build HTML for a line of weather metrics.
        
        :param metric_groups: List of metric group HTML strings
        :return: HTML string for the metrics line
        """
        if not metric_groups:
            return ""
        
        # Insert separators between metric groups
        result = []
        for i, group in enumerate(metric_groups):
            result.append(group)
            if i < len(metric_groups) - 1:
                result.append(self.build_separator())
        
        content = "".join(result)
        return f'<div class="weather-metrics-line">{content}</div>'


def get_style_manager() -> StyleManager:
    """
    Get the singleton StyleManager instance.
    
    :return: StyleManager instance
    """
    return StyleManager()