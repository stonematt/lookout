from ambient_api.ambientapi import AmbientAPI
import streamlit as st


# %%
# open Ambient Weather Network
AMBIENT_ENDPOINT = st.secrets["AMBIENT_ENDPOINT"]
AMBIENT_API_KEY = st.secrets["AMBIENT_API_KEY"]
AMBIENT_APPLICATION_KEY = st.secrets["AMBIENT_APPLICATION_KEY"]

# api = AmbientAPI()
api = AmbientAPI(
    log_level="INFO",
    AMBIENT_ENDPOINT=AMBIENT_ENDPOINT,
    AMBIENT_API_KEY=AMBIENT_API_KEY,
    AMBIENT_APPLICATION_KEY=AMBIENT_APPLICATION_KEY,
)

devices = api.get_devices()
