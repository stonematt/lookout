"""
Main streamlit.io application
"""

import time

import pandas as pd
import streamlit as st

import lookout.api.ambient_client as ambient_client
import lookout.core.data_processing as lo_dp
from lookout.ui import diagnostics, overview, playground, rain, rain_events
from lookout.utils.log_util import app_logger

# if st.secrets.get("DEBUG", False):
#     try:
#         import debugpy
#
#         # Avoid calling listen() multiple times
#         if not hasattr(debugpy, "_already_listening"):
#             debugpy.listen(("localhost", 5678))
#             debugpy._already_listening = True
#             print("🐛 Waiting for debugger to attach on port 5678...")
#             debugpy.wait_for_client()
#         else:
#             print("🐛 Debugger already listening.")
#     except Exception as e:
#         print(f"🐛 Failed to set up debugpy: {e}")


logger = app_logger(__name__)


def render_current_conditions_summary():
    """Render current conditions summary in the header row."""
    import json

    # Get current data
    last_data = st.session_state.get("last_data", {})

    # Initialize variables
    temp_f = last_data.get("tempf", 0)
    humidity = last_data.get("humidity", 0)
    barom_relin = last_data.get("baromrelin", 0)
    wind_speed = last_data.get("windspeedmph", 0)
    wind_dir = last_data.get("winddir", 0)
    uv = last_data.get("uv", 0)
    daily_rain = last_data.get("dailyrainin", 0)
    event_rain = last_data.get("eventrainin", 0)

    # Calculate trends (simplified - would need historical data for proper trends)
    temp_trend = "→"  # Neutral for now
    barom_trend = "→"  # Neutral for now

    # Format wind direction
    wind_dir_cardinal = "N"
    if 337.5 <= wind_dir or wind_dir < 22.5:
        wind_dir_cardinal = "N"
    elif 22.5 <= wind_dir < 67.5:
        wind_dir_cardinal = "NE"
    elif 67.5 <= wind_dir < 112.5:
        wind_dir_cardinal = "E"
    elif 112.5 <= wind_dir < 157.5:
        wind_dir_cardinal = "SE"
    elif 157.5 <= wind_dir < 202.5:
        wind_dir_cardinal = "S"
    elif 202.5 <= wind_dir < 247.5:
        wind_dir_cardinal = "SW"
    elif 247.5 <= wind_dir < 292.5:
        wind_dir_cardinal = "W"
    elif 292.5 <= wind_dir < 337.5:
        wind_dir_cardinal = "NW"

    # Format UV level
    uv_level = "Low"
    if uv >= 8:
        uv_level = "Very High"
    elif uv >= 6:
        uv_level = "High"
    elif uv >= 3:
        uv_level = "Moderate"

    # Calculate time since last rain
    rain_status = "Dry"
    time_since_rain = ""
    try:
        if last_data.get("lastRain"):
            last_rain = pd.to_datetime(last_data["lastRain"])
            current_time = pd.to_datetime(last_data["dateutc"], unit="ms", utc=True)
            time_since = current_time - last_rain
            hours_since = time_since.total_seconds() / 3600

            if hours_since < 1:
                time_since_rain = f"{time_since.total_seconds()/60:.0f}min"
            else:
                time_since_rain = f"{hours_since:.1f}h"

            if hours_since < 24:
                rain_status = f"Dry {time_since_rain}"
            else:
                rain_status = f"Dry {hours_since/24:.1f}d"
    except Exception:
        rain_status = "Dry"

    # Active event detection
    active_event_info = None
    try:
        if "device" in st.session_state and "history_df" in st.session_state:
            device = st.session_state["device"]
            device_mac = device["macAddress"]
            file_type = "parquet"

            from lookout.core.rain_events import RainEventCatalog

            catalog = RainEventCatalog(device_mac, file_type)

            # Try to get events from session or storage
            events_df = None
            if "rain_events_catalog" in st.session_state:
                events_df = st.session_state["rain_events_catalog"]
            elif catalog.catalog_exists():
                events_df = catalog.load_catalog()

            if events_df is not None and not events_df.empty:
                # Check for ongoing events
                ongoing_events = []
                for _, event in events_df.iterrows():
                    is_ongoing = False
                    if "ongoing" in event and event["ongoing"]:
                        is_ongoing = True
                    elif "flags" in event and event["flags"]:
                        flags = event["flags"]
                        if isinstance(flags, str):
                            flags = json.loads(flags)
                        is_ongoing = flags.get("ongoing", False) is True

                    if is_ongoing:
                        ongoing_events.append(event)

                # Additional validation: check current data eventrainin
                current_eventrainin = last_data.get("eventrainin", 0) or 0

                # Filter ongoing events to only those that match current data
                validated_ongoing_events = []
                for event in ongoing_events:
                    if current_eventrainin > 0:
                        validated_ongoing_events.append(event)
                    else:
                        logger.info(
                            f"Event {event.get('event_id', 'unknown')[:8]} marked ongoing in catalog but current eventrainin=0, skipping"
                        )

                if validated_ongoing_events:
                    # Get the most recent ongoing event
                    latest_event = max(
                        validated_ongoing_events, key=lambda x: x["start_time"]
                    )

                    # Calculate duration and format
                    duration_h = latest_event["duration_minutes"] / 60
                    total_rain = latest_event["total_rainfall"]

                    # Calculate rain rate (simplified)
                    rain_rate = 0
                    if duration_h > 0:
                        rain_rate = total_rain / duration_h

                    # Format duration string
                    if duration_h >= 24:
                        duration_str = f"{duration_h/24:.1f}d"
                    else:
                        duration_str = f"{duration_h:.1f}h"

                    # Format start time
                    start_time_str = "Unknown"
                    try:
                        start_time = pd.to_datetime(
                            latest_event["start_time"], unit="ms"
                        )
                        start_time_str = start_time.strftime("%b %d %-I:%M %p")
                    except Exception:
                        pass

                    active_event_info = {
                        "duration": duration_str,
                        "total_rain": total_rain,
                        "rain_rate": rain_rate,
                        "start_time": start_time_str,
                        "duration_h": duration_h,
                    }

    except ImportError:
        # Rain events not available
        pass
    except Exception as e:
        logger.warning(f"Error detecting active rain events: {e}")

    # Render current conditions summary in column 2
    if active_event_info:
        # Active event format
        st.caption(
            f"""
        🌡️ {temp_f:.0f}°F {temp_trend} • 💨 {wind_speed:.0f}mph {wind_dir_cardinal} • 🌧️ {rain_status}    
        **🌧️ ACTIVE EVENT ({active_event_info['duration']} running)**
        Total: {active_event_info['total_rain']:.2f}" • Rate: {active_event_info['rain_rate']:.2f}"/hr • Last rain: {time_since_rain} ago
        Started: {active_event_info['start_time']} • Duration ongoing
        """
        )
        st.caption(
            f"""
        🌊 {barom_relin:.2f}" {barom_trend} • 💧 {humidity:.0f}% • ☀️ {uv_level}
        """
        )
    else:
        # No active event format
        st.caption(
            f"""
        🌡️ {temp_f:.0f}°F {temp_trend} • 💨 {wind_speed:.0f}mph {wind_dir_cardinal} • 🌧️ {rain_status}
        """
        )
        st.caption(
            f"""
        🌊 {barom_relin:.2f}" {barom_trend} • 💧 {humidity:.0f}% • ☀️ {uv_level}
        """
        )


st.set_page_config(
    page_title="Weather Station Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

AMBIENT_ENDPOINT = st.secrets["AMBIENT_ENDPOINT"]
AMBIENT_API_KEY = st.secrets["AMBIENT_API_KEY"]
AMBIENT_APPLICATION_KEY = st.secrets["AMBIENT_APPLICATION_KEY"]


# %%
# define variables
sec_in_hour = 3600 * 1000
bucket = "lookout"
auto_refresh_min = 6  # minutes to wait for auto update
auto_refresh_max = 3 * 24 * 60  # 3 days in minutes


# Setup and get data ########################

if "devices" not in st.session_state:
    devices = ambient_client.get_devices()
    st.session_state["devices"] = devices
else:
    devices = st.session_state["devices"]

device = False
device_last_dateutc = 0
last_data = {}

device_menu = "98:CD:AC:22:0D:E5"
if len(devices) == 1:
    device = devices[0]
    device_menu = device["macAddress"]
    device_name = device["info"]["name"]
    st.session_state["last_data"] = device["lastData"]
    last_data = st.session_state["last_data"]
    st.session_state["device"] = device  # ← Add this
    logger.debug(f"One device found:  {device['info']['name']}")

    # Two-column header layout
    header_col1, header_col2 = st.columns([1, 1])
    with header_col1:
        st.header(f"Station: {device_name}")
    with header_col2:
        render_current_conditions_summary()

    # Compare device's last data UTC with the archive max dateutc
    device_last_dateutc = device["lastData"].get("dateutc")

# if we dont' get a device from ambient. blow up.
if not device:
    st.write("No connection to Ambient Network")
    exit()

file_type = "parquet"
device_mac = device_menu
hist_file = f"{device_mac}.{file_type}"
# lookout/98:CD:AC:22:0D:E5.json

# pause for ambient.
time.sleep(1)

# %%
auto_update = st.sidebar.checkbox("Auto-Update", value=True)


# Call the wrapper to load or update the weather station data
lo_dp.load_or_update_data(
    device=device,
    bucket=bucket,
    file_type=file_type,
    auto_update=auto_update,
    short_minutes=auto_refresh_min,
    long_minutes=auto_refresh_max,
)

# Access the updated history_df and max_dateutc
history_df = st.session_state["history_df"]
history_max_dateutc = st.session_state["history_max_dateutc"]

st.sidebar.write(f"History as of: {history_df.date.max()}")

history_age_h = lo_dp.get_human_readable_duration(
    device_last_dateutc, history_max_dateutc
)

# Display in the sidebar
st.sidebar.write(f"Archive is {history_age_h} old.")
# %%


# Present the dashboard ########################

tab_overview, tab_rain, tab_rain_events, tab_diagnostics, tab_playground = st.tabs(
    ["Overview", "Rain", "Rain Events", "Diagnostics", "Playground"]
)

with tab_overview:
    overview.render()

with tab_rain:
    rain.render()

with tab_rain_events:
    rain_events.render()

with tab_diagnostics:
    diagnostics.render()

with tab_playground:
    playground.render()
