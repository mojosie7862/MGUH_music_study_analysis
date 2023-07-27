import json
import time

f = r'NHC-OSPX_EPOCX_183950_20230719_181709.json'

with open(f) as file:
    data = json.load(file)
    for i in data['record_details']['markers']:
        for k,v in i['data'].items():
            print(k, v)
        marker_label = i['data']['label']
        marker_value = i['data']['value']

