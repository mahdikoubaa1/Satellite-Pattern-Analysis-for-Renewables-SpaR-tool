from datetime import datetime
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
import rasterio
from pyproj import Transformer
import numpy as np
from io import BytesIO
import os
from pathlib import Path
from tqdm import tqdm
import json

transformer = Transformer.from_crs("EPSG:4326","EPSG:32633")
client_id = 'sh-db96db01-5814-405f-a077-240ed8403ec3'
client_secret = 'M6D9tOHQN6xoiXRyrFXsEJ2ACppvMvs7'

client = BackendApplicationClient(client_id=client_id)
oauth=None
token_lifetime=None
token_time=None
def checkToken():
        global token_lifetime
        global token_time
        '''
        Check if the token is still valid, if not, creates a new one.
        Updates the token_time attribute to the current time and adds the oauth class attribute.
        '''
        if token_time is None or (datetime.now() - token_time).seconds >= token_lifetime:
            token_time = datetime.now()
            createToken()
            
def createToken():
        global token_lifetime
        global oauth
        '''
        Creates a new token. Updates the token_lifetime attribute by the value given in the token.

        Returns:
            OAuth2Session: session with the token.
        '''
        # Create a session
        client = BackendApplicationClient(client_id=client_id)
        oauth = OAuth2Session(client=client)

        # Get token for the session
        oauth.fetch_token(token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
                                  client_secret=client_secret, include_client_id=True)
        token_lifetime = oauth.token['expires_in'] #seconds till token expires
def catalogue(b):
  global oauth
  checkToken()
  data = {"collections": [
    "sentinel-2-l2a"
  ],
  "datetime": "2023-06-23T00:00:00Z/2023-09-26T23:59:59Z",
  "bbox": b,
  "limit": 100
  }

  url = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
  
  response = oauth.post(url, json=data)
  return response.content
def get_products(b):
    catal=catalogue(b)
    dict_str = catal.decode("UTF-8")
    json_catal=json.loads(dict_str)
    meta=[{'date':f['properties']['datetime'],'CC':f['properties']['eo:cloud_cover']} for f in json_catal['features'] if f['properties']['platform']=='sentinel-2b']
    meta.sort(key= lambda x :x['CC'])
    meta=meta[0:2]
    print(meta)
    evalscript_inp1="""["""
    for info in meta:
        evalscript_inp1+="""
        {bands: ["B01","B02","B03","B04","B05","B06","B07", "B08", "B8A","B09","B11","B12","dataMask"], units: "DN", mosaicking: "SIMPLE"},"""
    evalscript_inp1+="""
    ]"""
    evalscript_inp2= []
    for info in meta:
        evalscript_inp2.append({
            "dataFilter": {
              "timeRange": {
                "from": info['date'],
                "to": info['date']
              }
            },
            "type": "sentinel-2-l2a",
            "processing": {"harmonizeValues": "true"}
          })
    return evalscript_inp1,evalscript_inp2
def download(b,path=None,wh=None,from_date="2023-06-23T00:00:00Z",to_date="2023-09-26T23:59:59Z"):
  global oauth
  #inp1,inp2 =get_products(b)
  checkToken()
  if wh is None: wh=(np.asarray(transformer.transform(b[3],b[2]))-np.asarray(transformer.transform(b[1],b[0]))).astype('int')//10

  evalscript = """
//VERSION=3

function setup() {
  return {
    input: [
      {
        bands: ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"],      
        units: "DN",            
      }
    ],
    output: [
      {
        id: "default",
        bands: 13,
        sampleType: "UINT16",        
      },    
    ],
    mosaicking: "SIMPLE",
  };
}


function evaluatePixel(sample) {
  return [
    sample.B01,
    sample.B02,
    sample.B03,
    sample.B04,
    sample.B05,
    sample.B06,
    sample.B07,
    sample.B08,
    sample.B8A,
    sample.B09,
    sample.B11,
    sample.B11,
    sample.B12
  ];
}
"""
  request = {
  "input": {
    "bounds": {
      "bbox": b
    },
    "data": [
      {
        "dataFilter": {
          "timeRange": {
            "from": from_date,
            "to": to_date
          },
          "mosaickingOrder": "leastCC"
        },
        "type": "sentinel-2-l2a",
        "processing": {"harmonizeValues": "false"}
      }
    ]
  },
  "output": {
    "width": int(wh[0]),
    "height": int(wh[1]),
    "responses": [
      {
        "identifier": "default",
        "format": {
          "type": "image/tiff"
        }
      }
    ]
  },
  "evalscript": evalscript
}
  url = "https://sh.dataspace.copernicus.eu/api/v1/process"
  response = oauth.post(url, json=request)
  if response.status_code != 200:
      print(response.content)
      return False
  if (not path is None):
    f = open(path, "wb")
    f.write(response.content)
    return True
  else:
    return BytesIO(response.content)
  
def main():
  data_path='/mnt/9TB/koubaa/23_06_26_09'
  new_data_path='/mnt/9TB/koubaa/23_06_26_09_new'
  names = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
  names=names[4800::] #922+138
  n= tqdm(names)
  for raster_name in n:
    n.set_postfix_str(raster_name)
    with rasterio.open(f"{data_path}/{raster_name}") as currdisp:
      h= currdisp.bounds
      if(not download (h,f'{new_data_path}/{raster_name}',[1200,1200])): return
  


if __name__ == '__main__':
    main()