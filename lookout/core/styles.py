"""
Centralized style management for weather dashboard components.

This module provides a scalable CSS architecture following SOLID principles
for consistent styling across the application.
"""

import re
from dataclasses import dataclass
from typing import Dict, Optional

import streamlit as st

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
    current_conditions_bg: str = "#f8f9fa"
    current_conditions_border: str = "#e9ecef"

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

    _instance: Optional["StyleManager"] = None
    _initialized: bool = False

    def __new__(cls) -> "StyleManager":
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize StyleManager with default configuration."""
        if not hasattr(self, "_config"):
            self._config = StyleConfig()

    @property
    def config(self) -> StyleConfig:
        """Get current style configuration."""
        return self._config

    def inject_styles(self) -> None:
        """Inject global CSS styles, ensuring they're always available."""
        # Always inject styles to ensure they're available after reruns
        # This prevents style loss when components are redrawn after interactions
        css = self._generate_css()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
        logger.debug("CSS styles injected/reinjected")

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
            display: flex;
            align-items: center;
            gap: 0.2rem;
            white-space: nowrap;
        }}
        
        .metric-separator {{
            color: {self.config.separator_color};
            margin: 0 {self.config.separator_margin};
            font-weight: 500;
        }}
        
        .current-conditions {{
            background-color: {self.config.current_conditions_bg};
            border: 1px solid {self.config.current_conditions_border};
            border-radius: 6px;
            padding: 0.5rem 0.75rem;
            margin: 0.5rem 0;
            font-size: 0.85rem;
            line-height: 1.3;
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
        
        .event-line {{
            display: flex;
            align-items: flex-start;
            margin-bottom: 0.25rem;
        }}
        
        .emoji-bullet {{
            flex-shrink: 0;
            margin-right: 0.5rem;
            font-size: 1.0em;
        }}
        
        .event-content {{
            flex: 1;
        }}
        
        .metrics-line {{
            margin-left: 1.5rem; /* Align with event content */
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
            
            .current-conditions {{
                font-size: 0.75rem;
                padding: 0.4rem 0.6rem;
                line-height: 1.2;
                margin: 0.3rem 0;
            }}
            
            .event-line {{
                margin-bottom: 0.2rem;
            }}
            
            .emoji-bullet {{
                margin-right: 0.4rem;
                font-size: 0.9em;
            }}
            
            .metrics-line {{
                margin-left: 1.2rem;
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
            
            .current-conditions {{
                font-size: 0.7rem;
                padding: 0.3rem 0.5rem;
                margin: 0.2rem 0;
            }}
            
            .event-line {{
                margin-bottom: 0.15rem;
            }}
            
            .emoji-bullet {{
                margin-right: 0.3rem;
                font-size: 0.85em;
            }}
            
            .metrics-line {{
                margin-left: 1.0rem;
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
            
            .current-conditions {{
                font-size: 0.65rem;
                padding: 0.25rem 0.4rem;
            }}
            
            .event-line {{
                margin-bottom: 0.1rem;
            }}
            
            .emoji-bullet {{
                margin-right: 0.25rem;
                font-size: 0.8em;
            }}
            
            .metrics-line {{
                margin-left: 0.9rem;
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

    def build_metric_group(
        self, emoji: str, value: str, unit: str = "", trend: str = ""
    ) -> str:
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
