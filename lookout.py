from ambient_api.ambientapi import AmbientAPI
import time
#import lookout.json

#token = json.load(open("lookou.json"))
AMBIENT_ENDPOINT =  'https://api.ambientweather.net/v1'
AMBIENT_API_KEY =  '1220328acd7b407fb2e9fa350f3c7637b4ec8b8449554e5999bc95f2f99e7eec'
AMBIENT_APPLICATION_KEY =  'c87444e53af942a68978293c30ce834ba766695088d74ad499c154d94a271868'

api = AmbientAPI(AMBIENT_API_KEY=AMBIENT_API_KEY, AMBIENT_APPLICATION_KEY=AMBIENT_APPLICATION_KEY)

print(f"api key: {api.api_key}")

devices = api.get_devices()
