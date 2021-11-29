 import requests
from PIL import Image
import io
import os
import json

#resp = requests.post("http://localhost:5000", files={'file': open('ServicePic/NarutoHokage.png', 'rb')})
ref_size = 500

url = "http://localhost:5000"
data = {'ref-size': ref_size,
        'greyscale': 'Ye' } # noly "Yes" will give the geyscale picture
filename = 'ServicePic/Naruto.png'
file_output_name = "Naruto{0}.png".format(ref_size)

files = [
    ('file', open(filename, 'rb')),
    ('data', ('data', json.dumps(data), 'application/json')),
]

resp = requests.post(url, files=files)
print(resp.status_code)
#print(resp.content)
#print(resp.json())

Image.open(io.BytesIO(resp.content)).save(os.path.join('ServicePicOutput', file_output_name))
