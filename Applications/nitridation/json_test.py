import json
import numpy as np

with open('cases.json') as json_file:
    data = json.load(json_file)

print(data['G7']['recession']['std-dev'])