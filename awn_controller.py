# %%
from ambient_api.ambientapi import AmbientAPI
import time
from dateutil import parser
import pandas as pd
import plotly.express as px
import streamlit as st
import logging
import storj_df_s3 as sj
from datetime import datetime


# %%
AMBIENT_ENDPOINT = st.secrets["AMBIENT_ENDPOINT"]
AMBIENT_API_KEY = st.secrets["AMBIENT_API_KEY"]
AMBIENT_APPLICATION_KEY = st.secrets["AMBIENT_APPLICATION_KEY"]

api = AmbientAPI(
    # log_level="INFO",
    AMBIENT_ENDPOINT=AMBIENT_ENDPOINT,
    AMBIENT_API_KEY=AMBIENT_API_KEY,
    AMBIENT_APPLICATION_KEY=AMBIENT_APPLICATION_KEY,
)


# %%
def get_device_history_to_date(device, end_date=None, limit=288):
    history = []
    if end_date:
        # st.write(f"end date: {end_date} for history page")
        history = device.get_data(end_date=end_date, limit=limit)
    else:
        # st.write("no end date submitted")
        history = device.get_data(limit=limit)

    if history:
        return history
    else:
        st.write("someting broke")
        return history


# %%
def main():
    return True


if __name__ == "__main__":
    main()
# %%
