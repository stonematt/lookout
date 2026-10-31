from ambient_api.ambientapi import AmbientAPI
from lookout.secrets_loader import get_secret


# %%
# open Ambient Weather Network
AMBIENT_ENDPOINT = get_secret("AMBIENT_ENDPOINT")
AMBIENT_API_KEY = get_secret("AMBIENT_API_KEY")
AMBIENT_APPLICATION_KEY = get_secret("AMBIENT_APPLICATION_KEY")

# api = AmbientAPI()
api = AmbientAPI(
    log_level="INFO",
    AMBIENT_ENDPOINT=AMBIENT_ENDPOINT,
    AMBIENT_API_KEY=AMBIENT_API_KEY,
    AMBIENT_APPLICATION_KEY=AMBIENT_APPLICATION_KEY,
)

devices = api.get_devices()

# %%
