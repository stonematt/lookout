"""
Main streamlit.io application
"""

import time

import streamlit as st

import lookout.api.ambient_client as ambient_client
import lookout.api.awn_controller as awn
import lookout.core.data_processing as lo_dp
from lookout.ui import diagnostics, overview, playground, rain, rain_events
from lookout.ui import header
from lookout.core.styles import get_style_manager
from lookout.utils.log_util import app_logger
from lookout.utils.memory_utils import force_garbage_collection

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

# Initialize global styles
style_manager = get_style_manager()
style_manager.inject_styles()

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

    # Create header placeholder for dynamic updates
    header_placeholder = st.empty()
    st.session_state["header_placeholder"] = header_placeholder

    # Render weather header with current conditions
    with header_placeholder.container():
        header.render_weather_header(device_name)

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

if st.sidebar.button("🔄 Refresh Data"):
    try:
        awn.update_session_data(
            st.session_state["device"], st.session_state["history_df"]
        )
        st.sidebar.success("Data refreshed!")
    except Exception as e:
        st.sidebar.error("Failed to refresh data")


# Call the wrapper to load or update the weather station data
lo_dp.load_or_update_data(
    device=device,
    bucket=bucket,
    file_type=file_type,
    auto_update=auto_update,
    short_minutes=auto_refresh_min,
    long_minutes=auto_refresh_max,
)

# Access the updated history_df and timestamps
history_df = st.session_state["history_df"]
history_max_dateutc = st.session_state["history_max_dateutc"]
archive_max_dateutc = st.session_state.get("archive_max_dateutc", history_max_dateutc)

# Calculate current time for age calculations
current_time_utc = int(time.time() * 1000)

st.sidebar.write(f"History as of: {history_df.date.max()}")

# Archive age: how old the Storj archive data is (vs current time)
archive_age_h = lo_dp.get_human_readable_duration(current_time_utc, archive_max_dateutc)
st.sidebar.write(f"Archive is {archive_age_h} old")

# Last refresh: how fresh the most recent record in memory is (vs current time)
last_refresh_age_h = lo_dp.get_human_readable_duration(
    current_time_utc, history_max_dateutc
)
st.sidebar.write(f"Last refresh: {last_refresh_age_h} ago")

# Display current device data
st.sidebar.subheader("Current Data")
last_data = st.session_state.get("last_data", {})
st.sidebar.write(last_data)
# %%


# Present the dashboard ########################

# Check if we're in a dev environment
is_dev = st.secrets.get("environment", {}).get("dev", False)

# Create tab list based on environment
if is_dev:
    tab_names = ["Overview", "Rain", "Rain Events", "Diagnostics", "Playground"]
else:
    tab_names = ["Overview", "Rain", "Rain Events"]

tabs = st.tabs(tab_names)

# Map tab names to their corresponding render functions
tab_modules = {
    "Overview": overview,
    "Rain": rain,
    "Rain Events": rain_events,
    "Diagnostics": diagnostics,
    "Playground": playground,
}

# Render each tab with its corresponding module
for tab, name in zip(tabs, tab_names):
    with tab:
        tab_modules[name].render()

# Reduced frequency memory cleanup to prevent disruptive reruns
# Only cleanup on higher thresholds to avoid interfering with user interactions
if st.session_state.get("session_counter", 0) % 20 == 0:
    force_garbage_collection()
