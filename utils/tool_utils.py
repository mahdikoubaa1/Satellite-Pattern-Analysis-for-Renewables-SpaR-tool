import math
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import json
import pickle
from json2html import *
import sys
sys.path.append("../")
from models import SAM
import numpy as np
import os
from pathlib import Path
import pandas as pd
from io import StringIO, BytesIO
import plotly.graph_objects as go
from torch._C._onnx import TrainingMode
from matplotlib.colors import ListedColormap
import seaborn as sns
from pyproj import Transformer
import pytz
import rasterio
from osgeo import gdal
from scipy import ndimage as nd
import skimage.morphology as morph
from threading import Thread
import osmnx as ox
# ... and suppress errors
gdal.PushErrorHandler('CPLQuietErrorHandler')
import pycrs
from rasterio.features import rasterize
import yaml
from models.highres.transformers import *
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from models.highres.model import SolarPanelsModel
from owslib.wms import WebMapService
from rasterio.enums import Resampling
import shutil
from rasterio.mask import mask
from shapely.geometry import box
import random
from tqdm import tqdm
import torch
import gc
from split_tiff import Split
from rgb_js import rgbAdjustment
from skimage.transform import resize
import matplotlib.pyplot as plt
import geopandas as gpd
from geopandas.geoseries import GeoSeries
from shapely.geometry import shape
from collections import OrderedDict
from PIL import Image
from time import sleep
from ipyleaflet import Map, basemaps, basemap_to_tiles, GeoData,LayersControl,ImageOverlay,WidgetControl , GeomanDrawControl, FullScreenControl, Popup,WMSLayer, Rectangle
from ipyleaflet import Polygon as Poly
from shapely import MultiPoint, Point, MultiPolygon,LineString,Polygon
from shapely import centroid, area
from shapely.ops import transform
import faiss
faiss.cvar.distance_compute_blas_threshold=6000
from models.pixelwise_classification import pixelwise
from torchvision import  transforms
from datetime import datetime, timedelta
import cv2
import ipywidgets
from ipywidgets import Layout, HTML,interact
from IPython.display import display, update_display,DisplayHandle,DisplayObject
import rasterio.transform
from base64 import b64encode
from utils.download_raster import download
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
def extract_upsampled_bands(raster):
    band60m = raster.read(indexes=[1,10,11],out_shape=(3,int(raster.height / 6),int(raster.width / 6)),resampling=Resampling.bilinear).astype('float32')
    band20m = raster.read(indexes=[5,6,7,9,12,13],out_shape=(6,int(raster.height / 2),int(raster.width / 2)),resampling=Resampling.bilinear).astype('float32')
    band10m = raster.read(indexes=[2,3,4,8]).astype('float32')
    band60m= np.moveaxis(cv2.resize(np.moveaxis(band60m,0,2),(raster.width,raster.height)),2,0)
    band20m= np.moveaxis(cv2.resize(np.moveaxis(band20m,0,2),(raster.width,raster.height)),2,0)
    bands=np.concatenate((band60m[0].reshape(1,band60m.shape[1],band60m.shape[2]),band10m[(0,1,2),:,:],band20m[(0,1,2),:,:],band10m[3].reshape(1,band10m.shape[1],band10m.shape[2]),band20m[3].reshape(1,band20m.shape[1],band20m.shape[2]),band60m[(1,2),:,:],band20m[(4,5),:,:]),axis=0)
    return bands
def get_buffer_size(row, resolution):
        """
        Dynamically determine buffer size based on road type.
        """
        road_type = row["highway"]
        if isinstance(road_type, list):
            road_type = road_type[0]

        buffer_multipliers = {
            "motorway": 6,
            "trunk": 5,
            "primary": 4,
            "secondary": 3,
            "tertiary": 2,
            "residential": 1,
            "unclassified": 1,
            "service": 1,
        }
        multiplier = buffer_multipliers.get(road_type, 1)  
        return resolution * multiplier  

def getBoxTiff(poly,transform,height,width,resolution=0.00001):
    """
    Convert road network from OSMnx to a GeoTIFF file.
    """
    G = ox.graph_from_polygon(polygon=poly,trunc= False, network_type="drive")
    #print("Converting graph to GeoDataFrame...")
    _, edges = ox.graph_to_gdfs(G)
    edges['geometry']=edges['geometry'].intersection(poly)
    edges=edges[edges['geometry']!=LineString([])]
    #print("Rasterizing road network...")
    shapes = ((row.geometry.buffer(get_buffer_size(row, resolution)), 1) for _, row in edges.iterrows())
    if edges.size!=0:
        raster = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, dtype='uint8')
    else:
        raster=np.zeros([height,width])
    return 1-raster
class model :
    tool=None
    config_path=None
    configs=None
    pixnn=None
    widget=None
    g=None
    curr_pred=None
    highres_pred=None
    masks_predicted = GeoData(geo_dataframe = gpd.GeoDataFrame({"geometry" : [], "properties": []}),style={'color':'yellow','fillColor':'yellow','fillOpacity': 0.2},
                       hover_style={'fillColor': 'red' , 'fillOpacity': 0.2},
                       name = 'masks_predicted'
                       )
    #mean = [2279.1047, 2011.3107, 1923.4674, 1756.1125, 2057.2666, 3134.0662,
    #    3605.1943, 3540.2402, 3875.7695, 1767.3289, 1010.7678, 2826.0640,
    #    2066.8394,11.0551, 50.8178]
    #std = [180.2996, 240.9028, 288.8739, 438.9496, 419.7839, 677.7819, 873.3377,
    #    886.2139, 949.8568, 258.4645,   3.3687, 731.5524, 677.3787,1.9841, 1.8167]

    mean = [1358.8998, 1435.4919, 1668.5059, 1626.6582, 2070.2781, 3291.0283,
        3738.7976, 3874.6724, 3970.9172, 3997.7971, 3030.4189, 3030.4189,
        2305.4705]
    std = [223.0625, 308.9918, 365.0558, 520.7594, 509.7044, 741.4105, 923.0206,
        999.9840, 963.2914, 876.7068, 797.8113, 797.8113, 820.2186]
    
    #mean=[ 356.4907,  437.6278,  673.4810,  635.2744, 1080.7875, 2287.2554,
    #    2726.8997, 2859.0754, 2955.5269, 2982.0300, 2047.2684, 2047.2684,
    #    1321.8688]
    #std=[ 232.1855,  309.9370,  364.0717,  521.6251,  511.1423,  755.7710,
    #     937.8778, 1007.6766,  974.4764,  894.8656,  801.0314,  801.0314,
    #     822.1957]
    normalize=None
    models=None

    train = ipywidgets.Button(description="Train Model")
    savm = ipywidgets.Button(description="Save Model")
    training_loss=ipywidgets.Label(f"training loss:")
    validation_loss=ipywidgets.Label(f"validation loss:")
    train_and_save= ipywidgets.HBox([train,savm])
    setcutoff=ipywidgets.Button(description='Set Cutoff (km2)',layout=Layout(width='auto'))

    epochs_widget=None
    layers_widget=None
    select_input=None
    lr_widget=None
    gamma_widget=None
    csv_data=None
    long_lat_index=None
    def __init__(self, tool, config_path, tabular_data_path,device):
        self.tool=tool
        self.config_path=config_path
        self.configs = [f'{self.config_path}/{f}' for f in os.listdir(f'{self.config_path}') if os.path.isdir(os.path.join(f'{self.config_path}', f))]
        self.configs= sorted(self.configs, key= os.path.getmtime,reverse=True)
        self.configs=[f.split('/')[3] for f in self.configs]
        self.csv_data=pd.read_csv(tabular_data_path)
        self.csv_data.dropna(subset=['Longitude','Latitude'],inplace=True)
        self.device=device
        self.arr=np.asarray([self.tool.transformer.transform(i[1]['Latitude'],i[1]['Longitude']) for i in self.csv_data[['Longitude','Latitude']].iterrows()])
        

        self.highres_params={

            'architecture': 'Segformer',
            'encoder': 'mit_b5',
            'classes': ['solar_panel'],
            'lr': 0.0001,
            'epochs': 50,
            'batch_size': 16,

            'train_augmentation': get_training_augmentation,
            'valid_augmentation': get_validation_augmentation,
            'preprocessing': get_preprocessing,
            'num_workers': 2,

            'loss': smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True),
            'optimizer': torch.optim.Adam
        }
        self.segformer= SolarPanelsModel(
            arch=self.highres_params['architecture'],
            encoder=self.highres_params['encoder'],
            in_channels=3,
            out_classes=len(self.highres_params['classes']),
            model_params=self.highres_params
        )
        self.segformer.load_state_dict(torch.load('../data/segformer_mit_b5_e50.pth'), strict=False)
        self.segformer=self.segformer.to(self.device)
        with open('../data/ma_B_cuda.yaml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.SAMmodel = SAM.make(config['model']).to(self.device)
        self.SAMmodel.load_state_dict(torch.load('../data/model_epoch_best.pth', map_location=self.device),strict=True)
        self.long_lat_index  = faiss.IndexFlatL2(self.arr.shape[1])
        #res=faiss.StandardGpuResources()
        #self.long_lat_index = faiss.index_cpu_to_gpu(res,0,index=index_flataaaa )
        self.long_lat_index.add(self.arr)
        if (len (self.configs)!= 0):
            file = open(f'{self.config_path}/{self.configs[0]}/v_opts.pkl', 'rb')
            # dump information to that file
            self.tool.model_opts = pickle.load(file)
            # close the file
            file.close()
            self.pixnn= pixelwise(len(self.tool.model_opts['input_bands']),layers=[[int (x) for x in n.split(',')]  for n in self.tool.model_opts['layers'].split(';')],clayers=[[int (x) for x in n.split(',')]  for n in self.tool.model_opts['c_layers'].split(';')],clr=self.tool.model_opts['c_lr'],alpha=0.5, lr=self.tool.model_opts['lr'],gamma=self.tool.model_opts['gamma'])
            self.pixnn.load_state_dict(torch.load(f'{self.config_path}/{self.configs[0]}/pixnn.pt',map_location=self.device))
            self.pixnn.to(self.device)
        else:
            self.tool.model_opts= {'c_epochs': 4000,'epochs': 4000, 'input_bands':[i+1 for i in range(13)],'cutoff_area':(0.004),'points_per_polygon':10,'c_lr':1e-02,'lr':1e-02,'train_hist':[],'val_hist':[],'c_train_hist':[],'c_val_hist':[],'dataset':'','gamma': 2.,'c_layers':'16,0;16,0;5,0', 'layers':'512,2;256,2;128,2;16,0'}
            self.tool.model_opts["input_bands"].remove(11)


        trace_tr = go.Scatter(x=[i for i in range(1,len(self.tool.model_opts['train_hist'])+1)],y=self.tool.model_opts['train_hist'], opacity=0.75, name='Training Loss',mode='lines')
        trace_val = go.Scatter(x=[i for i in range(1,len(self.tool.model_opts['val_hist'])+1)],y=self.tool.model_opts['val_hist'], opacity=0.75, name='Validation Loss',mode='lines')
        trace_c_tr = go.Scatter(x=[i for i in range(1,len(self.tool.model_opts['c_train_hist'])+1)],y=self.tool.model_opts['c_train_hist'], opacity=0.75, name='CTraining Loss',mode='lines')
        trace_c_val = go.Scatter(x=[i for i in range(1,len(self.tool.model_opts['c_val_hist'])+1)],y=self.tool.model_opts['c_val_hist'], opacity=0.75, name='CValidation Loss',mode='lines')
        self.g = go.FigureWidget(data=[trace_tr,trace_val,trace_c_tr,trace_c_val],
                            layout=go.Layout(
                                xaxis={'title':'Epochs'},
                                yaxis={'title':'Loss'},
                                title=dict(
                                    text='Loss'
                                ),
                            ))
        self.masks_predicted.on_click(self.show_polygon_info)
        self.normalize= transforms.Normalize([self.mean[i-1]for i in self.tool.model_opts['input_bands']],[self.std[i-1]for i in self.tool.model_opts['input_bands']],inplace=True)
        self.tool.m1.add(self.masks_predicted)

        self.models=ipywidgets.Dropdown(options=self.configs,description='current config:',value =self.configs[0] if len(self.configs)!=0 else None, style={'description_width': 'auto'})
        
        self.cutoff=ipywidgets.FloatSlider(
            value=self.tool.model_opts['cutoff_area'],
            min=0,
            max=0.1,
            step=0.001,
            disabled=False,
            orientation='horizontal',
            readout=True,
            readout_format='.3f',
            layout=Layout(width='auto'),

        )
        self.setcutoff.on_click(self.set_cutoff)
        self.models.observe(self.change_model,'value')

        self.train.on_click(self.train_model)
        self.savm.on_click(self.save_model)
#        self.c_epochs_widget=ipywidgets.IntText(
#            value=self.tool.model_opts['c_epochs'],
#            description='c_epochs:',
#            disabled=False
#        )
#        self.c_layers_widget=ipywidgets.Text(
#            value=self.tool.model_opts['c_layers'],
#            description='c_layers:',
#            disabled=False
#        )
#        self.c_lr_widget=ipywidgets.FloatText(
#            value=self.tool.model_opts['c_lr'],
#            description='c_lr:',
#            disabled=False
#        )
        self.epochs_widget=ipywidgets.IntText(
            value=self.tool.model_opts['epochs'],
            description='Epochs:',
            disabled=False
        )
        self.layers_widget=ipywidgets.Text(
            value=self.tool.model_opts['layers'],
            description='Layers:',
            disabled=False
        )
        self.select_input=ipywidgets.SelectMultiple(
            options=[(i,i) for i in range(1,14)],
            value=self.tool.model_opts['input_bands'],
            rows=13,
            description='',
            disabled=False
        )
        self.lr_widget=ipywidgets.FloatText(
            value=self.tool.model_opts['lr'],
            description='Lr:',
            disabled=False
        )
        self.gamma_widget= ipywidgets.FloatText(
            value=self.tool.model_opts['gamma'],
            description='Gamma:',
            layout=Layout(width='140px'),
            disabled=False
        )
        self.widget=WidgetControl(widget=ipywidgets.Accordion(children=[ipywidgets.Tab([ipywidgets.VBox([self.models, ipywidgets.Label('input_features:'),self.select_input,self.epochs_widget,self.layers_widget,self.lr_widget, ipywidgets.Label('Focal Loss:'),self.gamma_widget,self.train_and_save]),ipywidgets.VBox([self.g,self.training_loss,self.validation_loss])],titles=['Hyperparameters', 'Loss']),ipywidgets.Tab([ipywidgets.VBox([self.cutoff,self.setcutoff],layout=Layout(width='400px'))],style={'description_width': 'auto'},titles=('Area Cutoff (S2 only)',))],titles=('Model Training (only for pixnn)','Postprocessing')), position='bottomleft')
        self.tool.m1.add(self.widget)
        self.tool.model=self
    def set_cutoff(self,b):
        
        self.tool.model_opts['cutoff_area']=self.cutoff.value
        if(not self.tool.model.pixnn is None):
            self.tool.model.masks_predicted.geo_dataframe= self.tool.model.curr_pred[self.tool.model.curr_pred["Area_synthetic"] > self.tool.model_opts['cutoff_area']]

            self.tool.model.masks_predicted._update_data('data')
    def trainpixnn(self,data,labels,epochs):
        lossf=self.pixnn.loss
        oneh=np.asarray([[1,0],[0,1]]) 
        ind= int(data.shape[0]*0.8)
        labels=oneh[labels]
        optimizer=self.pixnn.optimizer

        data=np.moveaxis(data,1,0)
    
        labels=np.moveaxis(labels,1,0)

        data= data.astype('float32').reshape((1,data.shape[0],-1,1))
        labels= labels.astype('float32').reshape((1,2,-1,1))
        tr_data= data[:,:,0:ind]
        val_data=data[:,:,ind:data.shape[2]]
        tr_labels= labels[:,:,0:ind]
        val_labels=labels[:,:,ind:data.shape[2]]
        tr_data = torch.from_numpy(tr_data)
        tr_labels=torch.from_numpy(tr_labels)
        val_data = torch.from_numpy(val_data)
        val_labels=torch.from_numpy(val_labels)
        tr_data=tr_data.to(self.device)
        tr_labels=tr_labels.to(self.device)
        val_data=val_data.to(self.device)
        val_labels=val_labels.to(self.device)
        running_tr_loss=0.
        running_val_loss=0.
        epochs_array=[]
        self.tool.model_opts['train_hist']=[]
        self.tool.model_opts['val_hist']=[]
        with self.g.batch_update():
            self.g.data[0].x=epochs_array
            self.g.data[0].y=self.tool.model_opts['train_hist']
            self.g.data[1].x=epochs_array
            self.g.data[1].y=self.tool.model_opts['val_hist']
        self.tool.prog.description=f'Training Model'
        self.tool.prog.max=epochs
    
        for i in range(epochs):
            self.training_loss.value=(f'training loss:{running_tr_loss}')
            self.validation_loss.value=(f'validation loss:{running_val_loss}')
            self.tool.prog.description=f'Training Model'

            self.tool.prog.value= i
            self.pixnn.train()
            optimizer.zero_grad()
            tr_predictions= self.pixnn(tr_data)
            tr_loss=lossf(tr_predictions,tr_labels)

            tr_loss.backward()
            optimizer.step()
            running_tr_loss-=running_tr_loss/(i+1)
            running_tr_loss+=tr_loss/(i+1)
            self.pixnn.eval()
            with torch.no_grad():
                val_predictions=self.pixnn(val_data)
                val_loss=lossf(val_predictions,val_labels)
                running_val_loss-=running_val_loss/(i+1)
                running_val_loss+=val_loss/(i+1)
            epochs_array.append(i)
            self.tool.model_opts['train_hist'].append(running_tr_loss.item())
            self.tool.model_opts['val_hist'].append(running_val_loss.item())

            if (i%500==0):
                with self.g.batch_update():
                    self.g.data[0].x=epochs_array
                    self.g.data[0].y=self.tool.model_opts['train_hist']
                    self.g.data[1].x=epochs_array
                    self.g.data[1].y=self.tool.model_opts['val_hist']
        with self.g.batch_update():
                self.g.data[0].x=epochs_array
                self.g.data[0].y=self.tool.model_opts['train_hist']
                self.g.data[1].x=epochs_array
                self.g.data[1].y=self.tool.model_opts['val_hist']
        self.tool.prog.description='Success!'
#    def traincontrastive(self,data,labels,epochs):
#        lossf=self.pixnn.cont.loss
#        oneh=np.asarray([[1,0],[0,1]])
#        labels=oneh[labels]
#        optimizer=self.pixnn.cont.optimizer
#
#        data=np.moveaxis(data,1,0)
#    
#        labels=np.moveaxis(labels,1,0)
#
#        data= data.astype('float32').reshape((1,data.shape[0],-1,1))
#        labels= labels.astype('float32').reshape((1,2,-1,1))
#        tr_data= data
#        tr_labels= labels
#        tr_data = torch.from_numpy(tr_data)
#        tr_labels=torch.from_numpy(tr_labels)
#       
#        tr_data=tr_data.to(self.device)
#        tr_labels=tr_labels.to(self.device)
#       
#        running_tr_loss=0.
#        epochs_array=[]
#        self.tool.model_opts['c_train_hist']=[]
#        with self.g.batch_update():
#            self.g.data[2].x=epochs_array
#            self.g.data[2].y=self.tool.model_opts['c_train_hist']
#        self.tool.prog.description=f'Training Model'
#        self.tool.prog.max=epochs
#    
#        for i in range(epochs):
#            
#
#            self.training_loss.value=(f'training loss:{running_tr_loss}')
#            self.tool.prog.description=f'Training Model'
#
#            self.tool.prog.value= i
#            self.pixnn.train()
#            optimizer.zero_grad()
#            tr_predictions= self.pixnn.cont(tr_data)
#            tr_loss=lossf(tr_predictions,tr_labels)
#
#            tr_loss.backward()
#            optimizer.step()
#            running_tr_loss-=running_tr_loss/(i+1)
#            running_tr_loss+=tr_loss/(i+1)
#            
#
#            epochs_array.append(i)
#            self.tool.model_opts['c_train_hist'].append(running_tr_loss.item())
#
#            if (i%500==0):
#                with self.g.batch_update():
#                    self.g.data[2].x=epochs_array
#                    self.g.data[2].y=self.tool.model_opts['c_train_hist']
#        with self.g.batch_update():
#                self.g.data[2].x=epochs_array
#                self.g.data[2].y=self.tool.model_opts['c_train_hist']
#        self.tool.prog.description='Success!'

    def predict_aerial(self,raster,poly,roadsmask):
        self.tool.prog.value=0
        self.tool.prog.description='Running Prediction'

        with torch.no_grad():

            img=(raster.read(indexes=[1,2,3]).astype('uint8'))
            if self.tool.visualizer.aerial_prediction_widget.value=='SAM' :
                trs=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
                    ])
                img=trs(np.moveaxis(img,0,2))[None,:,:,:]
                self.SAMmodel.eval()  
                img=img.to(device=self.device)
                a=torch.split(img,1024,2)
                a=[list(torch.split(f,1024,3)) for f in a]
            else:
                img=torch.Tensor(img).to(device=self.device)
                a=torch.split(img,1024,1)
                a=[list(torch.split(f,1024,2)) for f in a]
                self.segformer.eval()
            pred=[]
            for tx in a:
                x=[]
                for ty in tx:    
                    if 'SAM' in self.tool.visualizer.aerial_prediction_widget.value :
                        y=self.SAMmodel.infer(ty)
                    else:
                        y=self.segformer(ty)            
                    y = torch.sigmoid(y).cpu()
                    
                    y = (y.squeeze().numpy().round())
                    
                    x.append(y)
                pred.append(x)
            pred = [np.concatenate(tuple(x),axis=1) for x in pred ]
            pred = np.concatenate(tuple(pred),axis=0)
            pred = pred*roadsmask
            pred = nd.binary_closing(pred)
            pred = nd.binary_fill_holes(pred)
            pred = morph.erosion(pred, footprint=morph.disk(5))
            pred = morph.dilation(pred, footprint=morph.disk(5)).astype('uint16')
        shapes = rasterio.features.shapes(pred, transform=raster.transform)
        records = [{"geometry": shape(geometry)} for (geometry, value) in shapes if value == 1]
        if (len(records)!=0):
            geojson=gpd.GeoDataFrame(records, crs='EPSG:4326')
        else:
            geojson=gpd.GeoDataFrame({"geometry" : [], "properties": []}, crs='EPSG:4326')
        geojson['geometry']=geojson['geometry'].intersection(poly)
        geojson=geojson[geojson['geometry']!=Polygon([])]
        metr=geojson['geometry'].to_crs(32633)
        geojson['Area_synthetic']=metr.area/(10**6) 
        cent=geojson["geometry"].centroid
        geojson['Longitude_synthetic']=cent.x
        geojson['Latitude_synthetic']=cent.y
        geojson['Power_synthetic']=metr.area*0.18
        cent=cent.to_crs(32633)
        to_query=np.moveaxis( np.asarray([cent.x,cent.y]),0,1)
        distances, indices=self.long_lat_index.search(to_query,1)
        csventries=self.csv_data.iloc[indices.flatten()].reset_index()
        geojson=pd.concat([geojson,csventries],axis=1)
        self.highres_pred=geojson
        self.tool.prog.value=self.tool.prog.max
        self.tool.prog.description= 'Success!'
        #pred= self.yolov11(img)
        #plot= pred[0].plot()
        #plt.imshow(plot)
        #pred=pred[0].masks
        #if (not pred is None):
        #    pred= pred.data.cpu().detach().numpy()
        #    pred= resize(pred, (pred.shape[0],img.shape[0], img.shape[1]), anti_aliasing=True).astype('uint8')
        #    pred= pred.max(axis=0)
        #    shapes = rasterio.features.shapes(pred, transform=raster.transform)
        #    records = [{"geometry": shape(geometry)} for (geometry, value) in shapes if value == 1]
        #    geojson=gpd.GeoDataFrame(records, crs='EPSG:4326')
        #    metr=geojson['geometry'].to_crs(32633)
        #    geojson['Area_synthetic']=metr.area/(10**6) 
        #    cent=geojson["geometry"].centroid
        #    geojson['Longitude_synthetic']=cent.x
        #    geojson['Latitude_synthetic']=cent.y
        #    cent=cent.to_crs(32633)
        #    to_query=np.moveaxis( np.asarray([cent.x,cent.y]),0,1)
        #    distances, indices=self.long_lat_index.search(to_query,1)
        #    csventries=self.csv_data.iloc[indices.flatten()].reset_index()
        #    geojson=pd.concat([geojson,csventries],axis=1)
        #    self.highres_pred=geojson

    def predict_S2(self,image1,tiff):
        self.tool.prog.value=0
        self.tool.prog.description='Running Prediction'
        self.pixnn.eval()
        cols, rows = np.meshgrid(np.arange(image1.shape[2]), np.arange(image1.shape[1]))
        
        xs,ys=rasterio.transform.xy(self.tool.visualizer.transform1,rows,cols)
        pixcoors=np.array([xs,ys])
        image=np.concatenate((image1[[i for i in range(13)]],pixcoors), axis=0)
        image=image[[(i-1) for i in self.tool.model_opts['input_bands']]]
        image=image.reshape(1,image.shape[0],1,-1)
        image=torch.from_numpy(image.astype('float32'))
        image=image.to(self.device)
        if self.tool.sampling_opts['normalization']=='per raster' :
            m= torch.mean(image,(0,2,3))
            s= torch.std(image,(0,2,3))
            n=transforms.Normalize(m,s,inplace=True)
            n(image)
        else:
            self.normalize(image)
        image=image.detach()
        with torch.no_grad():
            image2= torch.split (image,1440000,dim=3)
            c1 = [self.pixnn(img) for img in image2 ]
            clustered=torch.cat(tuple(c1),dim=3)

            clustered=torch.softmax(clustered,dim=1)
            clustered= (clustered[:,1] >.95)
            #clustered= torch.argmax(clustered,dim=1)
        clustered=clustered.reshape(image1.shape[1],image1.shape[2])
        clustered =clustered.cpu().detach().numpy()
        pred=clustered.astype('uint8')
        shapes = rasterio.features.shapes(pred, transform=self.tool.visualizer.transform1)
        
        records = [{"geometry": shape(geometry)}
           for (geometry, value) in shapes if value == 1]
        if (len(records)!=0):
            geojson=gpd.GeoDataFrame(records, crs='EPSG:4326')
        else:
            geojson=gpd.GeoDataFrame({"geometry" : [], "properties": []}, crs='EPSG:4326')
        metr=geojson['geometry'].to_crs(32633)
        geojson['Area_synthetic']=metr.area/(10**6) 
        cent=geojson["geometry"].centroid
        geojson['Longitude_synthetic']=cent.x
        geojson['Latitude_synthetic']=cent.y
        geojson['Power_synthetic']=metr.area*(2/math.sqrt(3))*0.18
        cent=cent.to_crs(32633)
    
        to_query=np.moveaxis( np.asarray([cent.x,cent.y]),0,1)
        distances, indices=self.long_lat_index.search(to_query,1)
        csventries=self.csv_data.iloc[indices.flatten()].reset_index()
        geojson=pd.concat([geojson,csventries],axis=1)
        self.curr_pred=geojson
        self.tool.prog.value=self.tool.prog.max
        self.tool.prog.description= 'Success!'
    def show_polygon_info(self,event,feature, properties,id):
        if self.tool.sampler.sample_from_masks.value:
            self.tool.sampler.draw_control.newpolygon={'type':'Feature','geometry': feature['geometry'], 'properties':{'style': ( {'pane': 'overlayPane','source':self.tool.visualizer.raster_name, 'attribution': None, 'bubblingMouseEvents': True, 'fill': True, 'smoothFactor': 1, 'noClip': False, 'stroke': True, 'color': 'red', 'weight': 3, 'opacity': 1, 'lineCap': 'round', 'lineJoin': 'round', 'dashArray': None, 'dashOffset': None, 'fillColor': 'red', 'fillOpacity': 0.1, 'fillRule': 'evenodd', 'interactive': True, '_dashArray': None} if (self.tool.sampler.label.value=='solarfarm') else {'pane': 'overlayPane','source':self.tool.visualizer.raster_name, 'attribution': None, 'bubblingMouseEvents': True, 'fill': True, 'smoothFactor': 1, 'noClip': False, 'stroke': True, 'color': 'blue', 'weight': 3, 'opacity': 1, 'lineCap': 'round', 'lineJoin': 'round', 'dashArray': None, 'dashOffset': None, 'fillColor': 'blue', 'fillOpacity': 0.1, 'fillRule': 'evenodd', 'interactive': True, '_dashArray': None}),'label': ( 1 if (self.tool.sampler.label.value=='solarfarm') else 0)}}
            self.tool.sampler.draw_control.notify_change({'name':'newpolygon', 'type':'change'})
        else :
            htab=json2html.convert(json = properties)
            message1 = HTML()
            message1.value= f"""polygon metadata:
            {htab}
            """
            self.tool.popup.location=(properties['Latitude_synthetic'],properties['Longitude_synthetic'])
            self.tool.popup.child=message1
    def train_model(self,b):
        
        if (self.tool.sampler.dataset_widget!='unsaved_dataset'): self.tool.model_opts['dataset']=self.tool.sampler.dataset_widget.value
        self.tool.model_opts['input_bands']=self.select_input.value
        self.tool.model_opts['epochs']=self.epochs_widget.value
        #self.tool.model_opts['c_epochs']=self.c_epochs_widget.value
        self.tool.model_opts['lr']=self.lr_widget.value
        #self.tool.model_opts['c_lr']=self.c_lr_widget.value
        self.tool.model_opts['gamma']=self.gamma_widget.value
        #self.tool.model_opts['c_layers']=self.c_layers_widget.value
        self.tool.model_opts['layers']=self.layers_widget.value
        self.pixnn= pixelwise(len(self.tool.model_opts['input_bands']),layers=[[int (x) for x in n.split(',')]  for n in self.tool.model_opts['layers'].split(';')],clayers=[[int (x) for x in n.split(',')]  for n in self.tool.model_opts['c_layers'].split(';')],clr=self.tool.model_opts['c_lr'],alpha=0.5, lr=self.tool.model_opts['lr'],gamma=self.tool.model_opts['gamma'])
        self.pixnn=self.pixnn.to(self.device)
        if (b.description=="Train Model" and len(self.tool.sampler.draw_control.data)):
            cpts=self.tool.sampler.sampled_points.geo_dataframe.groupby('index').first()

            pts=self.tool.sampler.sampled_points.geo_dataframe.sample(frac=1)
            lr_data=np.array(pts[[f"{i}"for i in self.tool.model_opts['input_bands']]])
            lr_labels=np.array(pts['label'])
            clr_data=np.array(cpts[[f"{i}"for i in self.tool.model_opts['input_bands']]])
            clr_labels=np.array(cpts['label'])

            self.normalize.mean= [self.mean[i-1]for i in self.tool.model_opts['input_bands']]
            self.normalize.std= [self.std[i-1]for i in self.tool.model_opts['input_bands']]
            #self.traincontrastive(lr_data,lr_labels,self.tool.model_opts['c_epochs'])
            #self.pixnn.cont.requires_grad_(False)
            self.trainpixnn(lr_data,lr_labels,self.tool.model_opts['epochs'])
            if (not self.pixnn is None):
                self.predict_S2(self.tool.visualizer.currimg,self.tool.visualizer.currdisp)
                self.masks_predicted.geo_dataframe= self.curr_pred[self.curr_pred["Area_synthetic"] > self.tool.model_opts['cutoff_area']]
            self.masks_predicted._update_data('data')
        self.models.unobserve(self.change_model,'value')
        if (self.models.value is None or self.models._options_labels[0]!='unsaved_model'):
            l=list(self.models._options_labels)
            l.insert(0,'unsaved_model')
            self.models.options= l
        self.models.value='unsaved_model'
        self.models.observe(self.change_model,'value')
    def save_model(self,b):

        savepath = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
        os.makedirs(f'{self.config_path}/{savepath}')
        self.models.unobserve(self.change_model,'value')
        self.configs.insert(0,savepath)
        self.models.options=self.configs
        self.models.value=self.configs[0]

        torch.save(self.pixnn.state_dict(), f'{self.config_path}/{savepath}/pixnn.pt')
        inp=torch.randn((1, len(self.select_input.value), 1200, 1200), dtype=torch.float32).to(self.device)
        self.pixnn.eval()
        with torch.no_grad():
            torch.onnx.export(self.pixnn, (inp,),training=TrainingMode.TRAINING,f=f'{self.config_path}/{savepath}/pixnn.onnx') 
        file = open(f'{self.config_path}/{savepath}/v_opts.pkl', 'wb')
        pickle.dump(self.tool.model_opts, file)
        file.close()
        self.models.observe(self.change_model,'value')
    def change_model(self,change):


        if (self.models.value is None or self.models._options_labels[0]=='unsaved_model'):
            l=list(self.models._options_labels)
            l.remove('unsaved_model')
            self.models.options=l

        file = open(f'{self.config_path}/{change['new']}/v_opts.pkl', 'rb')
        # dump information to that file
        self.tool.model_opts = pickle.load(file)

        # close the file
        file.close()
        if (self.tool.model_opts['dataset'] in self.tool.sampler.datasets):
            self.tool.sampler.dataset_widget.value=self.tool.model_opts['dataset']
        self.pixnn= pixelwise(len(self.tool.model_opts['input_bands']),layers=[[int (x) for x in n.split(',')]  for n in self.tool.model_opts['layers'].split(';')],clayers=[[int (x) for x in n.split(',')]  for n in self.tool.model_opts['c_layers'].split(';')],clr=self.tool.model_opts['c_lr'],alpha=0.5, lr=self.tool.model_opts['lr'],gamma=self.tool.model_opts['gamma'])
        self.pixnn.load_state_dict(torch.load(f'{self.config_path}/{change['new']}/pixnn.pt',map_location=self.device))
        self.pixnn.to(self.device)

        self.epochs_widget.value=self.tool.model_opts['epochs']
        self.lr_widget.value=self.tool.model_opts['lr']
        self.layers_widget.value=self.tool.model_opts['layers']
#        self.c_epochs_widget.value=self.tool.model_opts['c_epochs']
#        self.c_lr_widget.value=self.tool.model_opts['c_lr']
#        self.c_layers_widget.value=self.tool.model_opts['c_layers']
        self.select_input.value=self.tool.model_opts['input_bands']
        self.tool.visualizer.cutoff.value=self.tool.model_opts['cutoff_area']
        self.gamma_widget.value=self.tool.model_opts['gamma']
        self.normalize.mean= [self.mean[i-1]for i in self.tool.model_opts['input_bands']]
        self.normalize.std= [self.std[i-1]for i in self.tool.model_opts['input_bands']]
        with self.g.batch_update():
                self.g.data[0].x=[i for i in range(1, len(self.tool.model_opts['train_hist'])+1)]
                self.g.data[0].y=self.tool.model_opts['train_hist']
                self.g.data[1].x=[i for i in range(1, len(self.tool.model_opts['val_hist'])+1)]
                self.g.data[1].y=self.tool.model_opts['val_hist']
                self.g.data[2].x=[i for i in range(1, len(self.tool.model_opts['c_train_hist'])+1)]
                self.g.data[2].y=self.tool.model_opts['c_train_hist']
                self.g.data[3].x=[i for i in range(1, len(self.tool.model_opts['c_val_hist'])+1)]
                self.g.data[3].y=self.tool.model_opts['c_val_hist']
        if(not self.pixnn is None):
            self.predict_S2(self.tool.visualizer.currimg,self.tool.visualizer.currdisp)
            self.masks_predicted.geo_dataframe= self.curr_pred[self.curr_pred["Area_synthetic"] > self.tool.model_opts['cutoff_area']]
            self.masks_predicted._update_data('data')
            
class sample:
    tool=None
    dataset_path=None
    datasets=None
    widget=None
    sampled_points=None
    draw_control=None
    dataset_widget=None
    label =ipywidgets.RadioButtons(
        options=['solarfarm', 'not solarfarm'],
        value='solarfarm',
        layout=Layout(width= 'max-content'), 
        disabled=False
    )
    normalization =None
    upsampling = None
    save = ipywidgets.Button(description="Sample and Save",style={'description_width': 'auto'},
        layout=Layout(height='36px'))
    s_vis= ipywidgets.Checkbox(
        value=True,
        description='',
        disabled=False,
        style={'description_width': 'auto'},
        layout=Layout(height='18px',margin='0 0 0 0',width='auto')
    )
    ns_vis= ipywidgets.Checkbox(
        value=True,
        description='',
        disabled=False,
        style={'description_width': 'auto'},
        layout=Layout(height='18px',margin='0 0 0 0',width='auto')
    )
    sample_from_masks= ipywidgets.ToggleButton(
        value=False,
        description='Sample from Mask',
        disabled=False,
        style={'description_width': 'auto'},
        layout=Layout(height='36px',margin='0 0 0 0')

    )
    sampling_weights_widgets=None

    def __init__(self, tool, dataset_path):
        self.tool=tool
        self.dataset_path=dataset_path
        self.datasets = [f'{self.dataset_path}/{f}' for f in os.listdir(f'{self.dataset_path}') if os.path.isdir(os.path.join(f'{self.dataset_path}', f))]
        self.datasets= sorted(self.datasets, key= os.path.getmtime,reverse=True)
        self.datasets=[f.split('/')[3] for f in self.datasets]
        self.dataset_widget=ipywidgets.Dropdown(options=self.datasets,description='dataset:',value =None, style={'description_width': 'auto'})
        polygon_sampling=None
        point_sampling=None
        if (self.tool.model_opts['dataset'] in self.datasets):
            self.dataset_widget.value=self.tool.model_opts['dataset']
        elif (len(self.datasets)!=0):
            self.dataset_widget.value=self.datasets[0]
        if (self.dataset_widget.value!=None):
            file = open(f'{self.dataset_path}/{self.dataset_widget.value}/d_opts.pkl', 'rb')
            self.tool.sampling_opts = pickle.load(file)
            polygon_sampling=gpd.read_file(f"{dataset_path}/{self.dataset_widget.value}/sampled_polygons.gpkg")
            point_sampling=gpd.read_file(f'{dataset_path}/{self.dataset_widget.value}/sampled_points.gpkg')
        else:
            self.tool.sampling_opts= {'upsampling':False,'normalization': 'global','sampling_weights':[60,60],'cindex':[]}
            polygon_sampling=gpd.GeoDataFrame({"geometry" : [], "properties": []})
            point_sampling=gpd.GeoDataFrame({"geometry" : [], "properties": []})
        self.sampled_points = GeoData (name='sampled_points',geo_dataframe = point_sampling,point_style={'radius': 5},hover_style={'fillColor': 'yellow' , 'fillOpacity': 0.8})
        self.sampled_points.on_click(self.show_info)
        self.tool.m1.add(self.sampled_points)
        self.draw_control = GeomanDrawControl(data=json.loads(polygon_sampling.to_json(drop_id=True))['features'],rotate=False)
        self.draw_control.circlemarker={}
        self.draw_control.polyline={}
        self.draw_control.edit=False
        self.draw_control.polygon = {"visibility": False, "pathOptions": {
                        "fillOpacity": 0.1,
                        "fillColor":'red',
                        "color":'red'
                        },
                        'continueDrawing':True}
        self.draw_polygon= ipywidgets.ToggleButton(
            value=False,
            description='Draw Polygons',
            disabled=False,
            style={'description_width': 'auto'},
            layout=Layout(height='36px')

        )
        self.draw_polygon.observe(self.sampling_draw,'value')
        self.draw_control.rectangle = {"visibility": False, "pathOptions": {
                        "fillOpacity": 0.1,
                        "fillColor":'green',
                        "color":'green'
                        }}
        self.draw_control.observe(self.change_mode,'current_mode')
        self.draw_control.observe(handler=self.handle_new_draw,names=['data'])
        self.tool.m1.add(self.draw_control)
        self.save.on_click(self.savepolygons)
        self.dataset_widget.observe(self.change_dataset,'value')
        self.ns_vis.observe(self.toggle_ns,names=['value'])
        self.s_vis.observe(self.toggle_s,names=['value'])
        self.sampling_weights_widgets=[ipywidgets.IntText(
            value=self.tool.sampling_opts['sampling_weights'][0],
            description='not_sf',
            layout=Layout(width='140px'),
            disabled=False
        ),ipywidgets.IntText(
            value=self.tool.sampling_opts['sampling_weights'][1],
            description='sf',
            layout=Layout(width='140px'),
            disabled=False
        )]
        self.normalization= ipywidgets.RadioButtons(
            options=['per raster', 'global'],
            value=self.tool.sampling_opts['normalization'],
            layout=Layout(width= 'max-content',direction='horizontal'), 
            disabled=False
        )
        self.upsampling = ipywidgets.Checkbox(
            value=self.tool.sampling_opts['upsampling'],
            description='Upsampling',
            disabled=False,
            style={'description_width': 'auto'},
            layout=Layout(margin='0 0 0 0')
        )
        
        self.label.observe(self.change_label,'value')

        self.widget = WidgetControl(widget=ipywidgets.Accordion(children=[ipywidgets.VBox([self.dataset_widget,ipywidgets.Label('label:'),ipywidgets.HBox([ HTML(
            """<style>
        .squaresf {
          height: 18px;
          width: 18px;
          background-color: blue;
        }
        .squarensf {
        height: 18px;
        width: 18px;
        background-color: red;
        }
        </style>
        <div class="squarensf"></div>
        <div class="squaresf"></div>"""),self.label,ipywidgets.VBox([self.s_vis,self.ns_vis]), self.sample_from_masks]),ipywidgets.Label('points per polygon:'),ipywidgets.HBox(self.sampling_weights_widgets),self.upsampling,self.draw_polygon,self.save])],titles=('Sampling (only for pixnn)',)), position='bottomright')

        self.tool.m1.add(self.widget)
        self.tool.sampler=self
    def change_mode(self,change):
        if (not change is None and change['new']!='draw:Polygon'):
            self.draw_polygon.unobserve(self.sampling_draw,'value')
            self.draw_polygon.value=False
            self.draw_polygon.observe(self.sampling_draw,'value')
        if (not change is None and change['new']!='draw:Rectangle'):
            
            self.tool.visualizer.download_S2.unobserve(self.tool.visualizer.S2_draw,'value')
            self.tool.visualizer.download_S2.value=False
            self.tool.visualizer.download_S2.observe(self.tool.visualizer.S2_draw,'value')
            self.tool.visualizer.download_aerial.unobserve(self.tool.visualizer.aerial_draw,'value')
            self.tool.visualizer.download_aerial.value=False
            self.tool.visualizer.download_aerial.observe(self.tool.visualizer.aerial_draw,'value')
    def sampling_draw(self,change):
        if change['new']:
            self.tool.sampler.draw_control.current_mode='draw:Polygon'
        else:
            self.tool.sampler.draw_control.current_mode=None
    def change_label (self,change):
        if change['new']=='solarfarm':
            if not self.s_vis.value:
                self.draw_control.polygon['pathOptions']['d']=True
            else:
                self.draw_control.polygon['pathOptions']['d']=False
            self.draw_control.polygon['pathOptions']['fillColor']='red'
            self.draw_control.polygon['pathOptions']['color']='red'
            self.draw_control.force=True
            self.draw_control.notify_change({'name':'polygon', 'type':'change'})
        else:
            if not self.ns_vis.value:
                self.draw_control.polygon['pathOptions']['d']=True
            else:
                self.draw_control.polygon['pathOptions']['d']=False
            self.draw_control.polygon['pathOptions']['fillColor']='blue'
            self.draw_control.polygon['pathOptions']['color']='blue'
            self.draw_control.force=True
            self.draw_control.notify_change({'name':'polygon', 'type':'change'})
    def toggle_s(self,change):
        self.draw_control.visible['s']=change['new']
        self.draw_control.force=True
        self.draw_control.notify_change({'name':'visible', 'type':'change'})
        if (self.label.value=='solarfarm'):
            if (not change['new']) :
                self.draw_control.polygon['pathOptions']['d']=True
                self.draw_polygon.value=False
                self.draw_polygon.disabled=True
            else:
                self.draw_control.polygon['pathOptions']['d']=False
                self.draw_polygon.disabled=False


            self.draw_control.force=True
            self.draw_control.notify_change({'name':'polygon', 'type':'change'})
        
    def toggle_ns(self,change):
        
        self.draw_control.visible['ns']=change['new']
        self.draw_control.force=True
        self.draw_control.notify_change({'name':'visible', 'type':'change'})
        if (self.label.value=='not solarfarm'):
            if (not change['new']) :
                self.draw_control.polygon['pathOptions']['d']=True
                self.draw_polygon.value=False
                self.draw_polygon.disabled=True
            else:
                self.draw_control.polygon['pathOptions']['d']=False
                self.draw_polygon.disabled=False

            self.draw_control.force=True
            self.draw_control.notify_change({'name':'polygon', 'type':'change'})

    def show_info(self,event,feature, properties,id):
        self.tool.popup.child=None
        message1 = HTML()
        message1.value= f'bands: {[properties[f'b{i}']for i in range(1,14)]}'
        self.tool.popup.location=(properties['15'],properties['14'])
        self.tool.popup.child=message1
    def savepolygons(self,b):
        
        self.tool.prog.value=0
        self.tool.prog.description='Saving downloaded rasters'
        for name,value in self.tool.visualizer.downloaded_rasters.items():
            if (value['counter']!=0):
                with open(f'{self.tool.visualizer.data_path}/{name}', "wb") as f:
                    f.write(value['bytes'].getbuffer()) 
        self.tool.visualizer.downloaded_rasters={} 
        self.tool.sampling_opts['sampling_weights']=[i.value for i in self.sampling_weights_widgets]
        self.tool.sampling_opts['upsampling']=self.upsampling.value
        self.tool.sampling_opts['normalization']=self.normalization.value
        self.tool.prog.description='Sampling'
        if (b.description=="Sample and Save" and len(self.draw_control.data)):
            savepath = datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
            os.makedirs(f'{self.dataset_path}/{savepath}')
            self.dataset_widget.unobserve(self.change_dataset,'value')
            self.datasets.insert(0,savepath)
            self.dataset_widget.options=self.datasets
            self.dataset_widget.value=self.datasets[0]
            s= self.draw_control.data
            
            s = sorted(s, key=lambda r: r['properties']['style']['source'])
            
            sampled_polygons= [{'geometry':shape(f['geometry']),'style':f['properties']['style'],'label':int(f['properties']['style']['color']=='red'), 'source':f['properties']['style']['source'] } for f in s if 'geometry' in f]
            sampled_polygons=gpd.GeoDataFrame(sampled_polygons)
            sampled_polygons.to_file(f"{self.dataset_path}/{savepath}/sampled_polygons.gpkg", driver='GeoJSON',crs=self.tool.visualizer.currdisp.crs.data)  

            
            sampled_polygons.loc[sampled_polygons['label']==1,'style']=[{'radius': 5, 'color': 'red', 'fillOpacity': 0.8, 'fillColor': 'red', 'weight': 3}for _ in range(sampled_polygons.loc[sampled_polygons['label']==1,'style'].count())]
            sampled_polygons.loc[sampled_polygons['label']==0,'style']=[{'radius': 5, 'color': 'blue', 'fillOpacity': 0.8, 'fillColor': 'blue', 'weight': 3}for _ in range(sampled_polygons.loc[sampled_polygons['label']==0,'style'].count())]
            sampled_polygons.loc[sampled_polygons['label']==1,'geometry']=sampled_polygons.loc[sampled_polygons['label']==1,'geometry'].sample_points(size=self.tool.sampling_opts['sampling_weights'][1])
            sampled_polygons.loc[sampled_polygons['label']==0,'geometry']=sampled_polygons.loc[sampled_polygons['label']==0,'geometry'].sample_points(size=self.tool.sampling_opts['sampling_weights'][0])
            
            self.sampled_points.geo_dataframe=sampled_polygons[['geometry','style','label','source' ]].explode(index_parts=False).reset_index(drop=False)
            #self.sampled_points.geo_dataframe=gpd.GeoDataFrame({"geometry" : sampled_polygons['geometry'].sample_points(size=opts['points_per_polygon']), "style": sampled_polygons['style'],'label':sampled_polygons['label'],'source':sampled_polygons['source']}).explode(ignore_index=True,index_parts=False)
            openforsampling=None
            image=None
            image1=None
            lastsrc=''
            m=None
            s=None
            bands=[]
            self.tool.prog.max=self.sampled_points.geo_dataframe.count()[0]
        
            for index,f in self.sampled_points.geo_dataframe.iterrows():
                
                self.tool.prog.value=index
                if (lastsrc!=f['source']):
                    lastsrc=f['source']
                    if (not openforsampling is None):
                        openforsampling.close()
                    openforsampling = rasterio.open(f"{self.tool.visualizer.data_path}/{lastsrc}")
                    if (self.tool.sampling_opts['upsampling']):                                    
                        image = torch.from_numpy(extract_upsampled_bands(openforsampling))
                    else: image= torch.from_numpy(openforsampling.read().astype('float32'))
                    image = image.to(self.tool.model.device)
                    m=torch.mean(image,(1,2))
                    s=torch.std(image,(1,2))
                    #aa=image.reshape([image.shape[0],-1])
                    #mi=torch.min (aa,1).values
                    #ma=torch.max (aa,1).values
                    #print (mi)
                    #print (ma)
                    #m= (mi+ma)/2
                    #s =(ma-mi)/2
                    if (self.tool.sampling_opts['normalization']=='per raster'):
                        n=transforms.Normalize(m,s)
                        image1=n(image)
                    else:
                        n=transforms.Normalize(self.tool.model.mean[:13],self.tool.model.std[:13])
                        image1=n(image)


                    image1=image1.detach()

                transform = openforsampling.transform
                p = f['geometry']
                p=~transform*(p.x,p.y)
                p= (int(p[0]),int(p[1]))
                to_append={f"{i}":image1[i-1,p[1],p[0]].item() for i in range(1,14)}
                to_append['14']=float(f['geometry'].x)
                to_append['15']=float(f['geometry'].y)
                to_append.update({f"b{i}":image[i-1,p[1],p[0]].item() for i in range(1,14)})
                bands.append(to_append)
            self.sampled_points.geo_dataframe[[f"{i}"for i in range(1,16)]+[f"b{i}"for i in range(1,14)]]=gpd.GeoDataFrame(bands)
            self.sampled_points._update_data('data')
            self.tool.prog.description='Saving'
            self.sampled_points.geo_dataframe.to_file(f"{self.dataset_path}/{savepath}/sampled_points.gpkg", driver='GeoJSON',crs=self.tool.visualizer.currdisp.crs.data)
            file = open(f'{self.dataset_path}/{savepath}/d_opts.pkl', 'wb')
            pickle.dump(self.tool.sampling_opts, file)
            file.close()

            self.tool.prog.description='Succcess!'
            self.dataset_widget.observe(self.change_dataset,'value')

    def handle_new_draw(self,change):

    
        self.dataset_widget.unobserve(self.change_dataset,'value')
        if (self.dataset_widget.value is None or self.dataset_widget._options_labels[0]!='unsaved_dataset'):
            l=list(self.dataset_widget._options_labels)
            l.insert(0,'unsaved_dataset')
            self.dataset_widget.options= l
        self.dataset_widget.value='unsaved_dataset'
        self.dataset_widget.observe(self.change_dataset,'value') 
    def change_dataset(self,change):
        file = open(f'{self.dataset_path}/{change['new']}/d_opts.pkl', 'rb')
        self.tool.sampling_opts = pickle.load(file)
        
        if (self.dataset_widget.value is None or self.dataset_widget._options_labels[0]=='unsaved_dataset'):
            l=list(self.dataset_widget._options_labels)
            l.remove('unsaved_dataset')
            self.dataset_widget.options=l
        self.tool.prog.value=0
        self.tool.prog.description= 'Dataset Loading:'
        self.draw_control.observe(handler=self.handle_new_draw,names=['data'])
        self.draw_control.unobserve(handler=self.handle_new_draw,names=['data'])

        self.draw_control.data=json.loads(gpd.read_file(f"{self.dataset_path}/{self.dataset_widget.value}/sampled_polygons.gpkg").to_json(drop_id=True))['features']
        self.draw_control.observe(handler=self.handle_new_draw,names=['data'])

        self.sampled_points.geo_dataframe=gpd.read_file(f'{self.dataset_path}/{self.dataset_widget.value}/sampled_points.gpkg')
        self.sampling_weights_widgets[0].value=self.tool.sampling_opts['sampling_weights'][0]
        self.sampling_weights_widgets[1].value=self.tool.sampling_opts['sampling_weights'][1]
        self.normalization.value= self.tool.sampling_opts['normalization']
        self.upsampling.value= self.tool.sampling_opts['upsampling']
        self.tool.prog.value=self.tool.prog.max
        self.tool.prog.description= 'Success!'
class visualize:
    tool=None
    data_path=None
    data=None
    widget=None
    currind=-1
    index1_gpu=None
    raster_name=''
    currdisp = None
    currimg = None
    t2rgb = rgbAdjustment()
    image2=ImageOverlay(url='', name='raster')
    transform1=None
    rastercenters=None
    previous = ipywidgets.Button(description="Previous S2 Raster",
            style={'description_width': 'auto'},
            layout=Layout(height='36px',width='auto'))
    next = ipywidgets.Button(description="Next S2 Raster",
            style={'description_width': 'auto'},
            layout=Layout(height='36px',width='auto'))
    closest = ipywidgets.Button(description="Closest S2 Raster",
            style={'description_width': 'auto'},
            layout=Layout(height='36px',width='auto'))
    save_pred = ipywidgets.Button(description="Save as GEOJSON",
            style={'description_width': 'auto'},
            layout=Layout(height='36px',width='auto'))
    cutoff=None
    cutoff_widget=None
    centcoors=None
    jump_to_widget=None
    resample_raster=None
    save_as_png=None
    rgb=None
    downloaded_rasters={}
    def __init__(self, tool, data_path):
        plt.rcParams.update({'font.size': 20})

        self.tool=tool
        self.data_path=data_path
        self.data = [f.replace(".tiff","").split("_") for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
        self.rastercenters= [np.array(self.tool.transformer.transform(float(d[1]),float(d[2]))) for d in self.data]
        self.rastercenters=np.array(self.rastercenters)
        self.index_flat = faiss.IndexFlatL2(self.rastercenters.shape[1])
        #self.index1_gpu = faiss.index_cpu_to_gpus_list(gpus=[gpu],index=self.index_flat )
        self.index_flat.add(self.rastercenters)
        self.bayernmaps = WebMapService('https://geoservices.bayern.de/od/wms/histdop/v1/histdop?')
        self.bayernlayers = [l for l in self.bayernmaps.contents.keys() if l[0] == 'b' and l[-1]=='h']
        self.bayernlayers_widget = ipywidgets.Dropdown(options=self.bayernlayers,description='Bayern:',value =self.bayernlayers[0], style={'description_width': 'auto'},layout=Layout(width='auto'))
        self.aerial_prediction_widget=ipywidgets.Dropdown(options=['SAM','Segformer'],description='Predict using:',value ='SAM', style={'description_width': 'auto'},layout=Layout(width='auto'))
        self.historic_aerial_bayern = WMSLayer(
            url='https://geoservices.bayern.de/od/wms/histdop/v1/histdop?',
            layers=self.bayernlayers_widget.value,
            format='image/png',
            crs={'name': 'EPSG4326', 'custom': False},
            transparent=True,
            attribution='BVV - geodaten.bayern.de',
            min_zoom = 0,
            max_zoom = 40,
            min_native_zoom = 0,
            max_native_zoom = 40
        )
        
        self.bayernlayers_widget.observe(self.update_wms,'value')
        self.tool.m1.add(self.historic_aerial_bayern)
        self.tool.m1.add(self.image2)
        self.from_date= ipywidgets.DatetimePicker(
            description='From:',
            value=datetime(2023,6,23,0,0,0,tzinfo=pytz.timezone('Etc/Zulu')),
            layout=Layout(width='auto'),
            style={'description_width': 'auto'},
            disabled=False
        )
        self.to_date= ipywidgets.DatetimePicker(
            description='To:',
            value=datetime(2023,9,26,23,59,59,tzinfo=pytz.timezone('Etc/Zulu')),
            layout=Layout(width='auto'),
            style={'description_width': 'auto'},
            disabled=False

        )
        self.centcoors= [ipywidgets.FloatText(
            value=self.tool.m1.center[0],
            description='Lat:',
            layout=Layout(width='150px'),
            style={'description_width': '40px'},

            disabled=False
        ),ipywidgets.FloatText(
            value=self.tool.m1.center[1],
            description='Long:',
            layout=Layout(width='150px'),
            style={'description_width': '40px'},

            disabled=False
        ),ipywidgets.Button(description='Jump')]
        self.centcoors[2].on_click(self.jump_to)
        self.resample_raster=ipywidgets.Button(description='Resample S2 Raster',
            style={'description_width': 'auto'},
            layout=Layout(height='36px',width='auto'))
        self.save_as_png= ipywidgets.Button(description='Save as PNG',
            style={'description_width': 'auto'},
            layout=Layout(height='36px',width='auto'))
        self.highresbounds= Rectangle(bounds=((1,1),(0,0)),color="green",fill_color="green",opacity=0,fill_opacity=0,totalenergy=None,totalarea=None)
        
        self.tool.m1.add(self.highresbounds)
        self.resample_raster.on_click(self.resampling)
        self.save_as_png.on_click(self.png_save)
        self.jump_to_widget=ipywidgets.VBox(children= self.centcoors)
        self.previous.on_click(self.vis)
        self.next.on_click(self.vis)
        self.closest.on_click(self.get_closest)
        self.save_pred.on_click(self.save_predictions)
        self.download_S2= ipywidgets.ToggleButton(
            value=False,
            description='Download S2 Raster',
            disabled=False,
            style={'description_width': 'auto'},
            layout=Layout(height='36px',width='auto',margin='0 0 0 0')

        )
        self.download_aerial= ipywidgets.ToggleButton(
            value=False,
            description='Download Aerial Raster',
            disabled=False,
            style={'description_width': 'auto'},
            layout=Layout(height='36px',width='auto',margin='0 0 0 0')

        )
    
        self.tool.m1.observe(self.update_coordinates,'center')
        self.download_aerial.observe(self.aerial_draw,'value')
        self.download_S2.observe(self.S2_draw,'value')
        self.S2_prediction_widget=ipywidgets.Dropdown(options=['PIXNN'],description='Predict using:',value ='PIXNN', style={'description_width': 'auto'},layout=Layout(width='auto'))
        S2download= ipywidgets.VBox([self.from_date,self.to_date,self.S2_prediction_widget,self.download_S2])
        S2local=ipywidgets.VBox([self.previous,self.next,self.closest,self.resample_raster,self.S2_prediction_widget])
        aerialdownload=ipywidgets.VBox([self.bayernlayers_widget,self.aerial_prediction_widget,self.download_aerial])
        #self.widget = WidgetControl(widget=ipywidgets.Accordion(children=[self.jump_to_widget,self.previous,self.next,self.closest,self.resample_raster,self.save_pred,self.save_as_png,self.cutoff,self.setcutoff, self.from_date,self.to_date,self.bayernlayers_widget,self.prediction_widget])],titles=('visualization',)), position='topright')
        self.widget = WidgetControl(widget=ipywidgets.Accordion(children=[self.jump_to_widget,ipywidgets.Tab([S2local,S2download,aerialdownload],layout=Layout(width='400px'),titles=('Local S2','Download S2','Download Aerial')),ipywidgets.VBox([self.save_as_png,self.save_pred])],titles=('Localization','Rasters','Save Predictions')), position='topright')
        #self.cutoff_widget = WidgetControl(widget=ipywidgets.VBox([ipywidgets.Label('Cutoff area in km²:'),self.cutoff,self.setcutoff]), position='topright')
        self.tool.m1.add(self.widget)
        self.tool.sampler.draw_control.on_draw(self.checkdraw)
        self.tool.visualizer=self

        self.vis(self.next)
    def aerial_draw(self, change):
        if change['new']:
            self.download_S2.unobserve(self.S2_draw,'value')
            self.download_S2.value=False
            self.download_S2.observe(self.S2_draw,'value')
            self.tool.sampler.draw_control.current_mode='draw:Rectangle'
        else:
            self.tool.sampler.draw_control.current_mode=None
    def S2_draw(self, change):
        if change['new']:
            self.download_aerial.unobserve(self.aerial_draw,'value')
            self.download_aerial.value=False
            self.download_aerial.observe(self.aerial_draw,'value')
            self.tool.sampler.draw_control.current_mode='draw:Rectangle'
        else:
            self.tool.sampler.draw_control.current_mode=None
    def update_wms(self,change):
        self.historic_aerial_bayern.layers=change['new']
        
    def png_save(self,b):
        
        shapes = list(filter(None, self.tool.model.masks_predicted.geo_dataframe["geometry"]))
        if (len (shapes)==0): mask1 = np.zeros((self.rgb.shape[0],self.rgb.shape[1]))
        else:
            mask1, _ = mask(self.currdisp, shapes, crop=False)
            mask1 = np.where(mask1!= 0, 1., mask1)
            mask1 = mask1[0]
        flatui = ["#3498db", "#FFD700"]

        color_map = ListedColormap(sns.color_palette(flatui).as_hex())
        image = resize(self.rgb, (self.rgb.shape[0], self.rgb.shape[1], 3), anti_aliasing=True)
        pred = resize(mask1, (self.rgb.shape[0], self.rgb.shape[1]), anti_aliasing=True)

        fig = plt.figure(figsize=(20, 20))
        #flatui = ["#3498db", "#651FFF"]
        plt.imshow(image)
        pred = np.ma.masked_where(pred == 0, pred)
        plt.imshow(pred, alpha=0.5, interpolation='none', cmap=color_map, vmin=0, vmax=1)
        plt.axis("off")
        plt.title(self.raster_name)
        plt.tight_layout()
        plt.savefig(f'{self.data_path}/png_predictions/{self.raster_name.split('/')[-1].replace('.tiff','.png')}')
        plt.close(fig)

    def resampling(self,b):
        if(self.raster_name[0]!='n'):
            self.image2.visible=True
            self.highresbounds.totalenergy=None

            self.highresbounds.opacity=self.highresbounds.fill_opacity=0

            self.tool.prog.value=0
            self.tool.prog.description='copying old raster'
            shutil.copy(f'{self.data_path}/{self.raster_name}',f'{self.data_path}/corrupted')
            self.tool.prog.description='downloading raster'
            h= self.currdisp.bounds
            download (h,f'{self.data_path}/{self.raster_name}',[1200,1200] )
            self.tool.prog.description='loading raster'
            if (not self.currdisp is  None):
                self.currdisp.close()
            self.currdisp = rasterio.open(f"{self.data_path}/{self.raster_name}")
            self.transform1 = self.currdisp.transform
            if self.tool.sampling_opts['upsampling']:
                self.currimg = extract_upsampled_bands(self.currdisp)
            else:
                self.currimg = self.currdisp.read().astype('float32')
            self.rgb= np.asarray(self.t2rgb.evaluate_pixel(self.currimg[3]/10000,self.currimg[2]/10000,self.currimg[1]/10000))
            vis_img=Image.fromarray(self.rgb)
            buffer=BytesIO()
            vis_img.save(buffer,'jpeg')
            dt=b64encode(buffer.getvalue())
            dt = dt.decode("ascii")
            self.image2.url="data:image/jpeg;base64,"+dt
            self.image2.bounds=((self.currdisp.bounds[1], self.currdisp.bounds[0]), (self.currdisp.bounds[3], self.currdisp.bounds[2]))
            self.tool.sampler.draw_control.polygon['pathOptions']['source']=self.raster_name
            self.tool.sampler.draw_control.polygon['pathOptions']['bounds']=self.image2.bounds
            self.tool.sampler.draw_control.force=True
            self.tool.sampler.draw_control.notify_change({'name':'polygon', 'type':'change'})
            #masks_saved.geo_dataframe=in_file
            if (not self.tool.model.pixnn is None):
                self.tool.model.predict_S2(self.currimg,self.currdisp)
                self.tool.model.masks_predicted.geo_dataframe= self.tool.model.curr_pred[self.tool.model.curr_pred["Area_synthetic"] > self.tool.model_opts['cutoff_area']]
                self.tool.model.masks_predicted._update_data('data')
            #masks_saved._update_data('data')
            self.tool.prog.value=self.tool.prog.max
            self.tool.prog.description='Success!'
    def show_box_info(self,event,coordinates,type):
        h=self.highresbounds.locations
        self.tool.popup.location=((h[0][0]+h[1][0])/2,(h[1][1]+h[2][1])/2)
        self.tool.popup.child=HTML(f"""Total_Power_Synthetic (kWp): {self.highresbounds.totalenergy}
        <br>
        Total_Area_Synthetic (km2): {self.highresbounds.totalarea}""")
    def checkdraw(self,*args, **kwargs):
        if (kwargs['action']=='create' and kwargs['geo_json'][0]['properties']['type']=='rectangle'): 
            
            h= kwargs['geo_json'][0]['geometry']['coordinates'][0]
            self.tool.m1.center=(float((h[0][1]+h[2][1])/2), float((h[0][0]+h[2][0])/2))
            self.tool.prog.value=0
            self.tool.prog.description='downloading raster'
            if self.download_aerial.value:
                self.download_aerial.value=False
                self.image2.visible=False
                self.highresbounds.opacity=1
                longres=2.20481e-06
                latres=1.38735e-06
                bbox=[h[0][0],h[0][1],h[2][0],h[2][1]]

                self.highresbounds.locations=[(bbox[1],bbox[0]),(bbox[3],bbox[0]),(bbox[3],bbox[2]),(bbox[1],bbox[2])]
                
                self.raster_name= self.raster_name=f'new/e_{float((h[0][1]+h[2][1])/2)}_{float((h[0][0]+h[2][0])/2)}_aerial_{self.bayernlayers_widget.value}.tiff'

                cropmask=[box(h[0][0],h[0][1],h[2][0],h[2][1])]
                bbox[2]+= (1024-((bbox[2]-bbox[0])/longres)%1024)*longres
                bbox[3]+= (1024-((bbox[3]-bbox[1])/latres)%1024)*latres
                height=round((bbox[3]-bbox[1])/latres)
                width=round((bbox[2]-bbox[0])/longres)
                img=self.bayernmaps.getmap(layers=[self.bayernlayers_widget.value],
                              styles=['default'],
                              srs='EPSG:4326',
                              bbox=bbox,
                              size=(width,height),
                              format='image/tiff',
                              transparent=True)
                self.tool.prog.value=self.tool.prog.max/2
                self.tool.prog.description='loading raster'
                buffer=BytesIO(img.read())
                raster=rasterio.open(buffer)
                roadmask=getBoxTiff(poly=Polygon(((bbox[0],bbox[1]),(bbox[0],bbox[3]),(bbox[2],bbox[3]),(bbox[2],bbox[1]),(bbox[0],bbox[1]))),transform=raster.transform,height=height,width=width)
                with open(f'{self.data_path}/{self.raster_name}', "wb") as f:
                    f.write(img.read()) 
                self.currimg, currtransform = mask(dataset=raster, shapes=cropmask, crop=True)
                self.rgb=np.moveaxis(self.currimg[0:3],0,2)
                epsg_code = 4326
                curr_meta = raster.meta.copy()

                curr_meta.update({"driver": "GTiff",
                     "height": self.currimg.shape[1],
                     "width": self.currimg.shape[2],
                     "transform": currtransform,
                     "crs": pycrs.parse.from_epsg_code(epsg_code).to_proj4()}
                             )
                writebuffer=BytesIO()
                with rasterio.open(writebuffer, "w", **curr_meta) as dest:
                    dest.write(self.currimg)
                self.downloaded_rasters[self.raster_name]={'counter':0,'bytes':writebuffer}
                self.currdisp= rasterio.open(writebuffer)
                self.tool.model.predict_aerial(raster,Polygon(h),roadmask)
                self.tool.model.masks_predicted.geo_dataframe= self.tool.model.highres_pred
                self.tool.model.masks_predicted._update_data('data')
                self.tool.sampler.draw_control.polygon['pathOptions']['bounds']=((1,1),(0,0))
                self.tool.sampler.draw_control.force=True
                self.tool.sampler.draw_control.notify_change({'name':'polygon', 'type':'change'})
                self.highresbounds.totalenergy=self.tool.model.highres_pred["Power_synthetic"].sum()
                self.highresbounds.totalarea=self.tool.model.highres_pred["Area_synthetic"].sum()
                self.highresbounds.on_click(self.show_box_info)

            else:
                self.download_S2.value=False
                self.image2.visible=True
                self.highresbounds.opacity=0

                if ((self.to_date.value - self.from_date.value).days<5):
                    self.from_date.value =self.to_date.value- timedelta(days=5)
                self.raster_name=f'new/e_{float((h[0][1]+h[2][1])/2)}_{float((h[0][0]+h[2][0])/2)}_{self.from_date.value.strftime('%Y-%m-%dT%H:%M:%SZ')}_{self.to_date.value.strftime('%Y-%m-%dT%H:%M:%SZ')}.tiff'
                in_mem= download (h[0]+h[2],from_date=self.from_date.value.strftime('%Y-%m-%dT%H:%M:%SZ')  ,to_date=self.to_date.value.strftime('%Y-%m-%dT%H:%M:%SZ'))
                self.downloaded_rasters[self.raster_name]={'counter':0,'bytes':in_mem}
                self.tool.prog.value=self.tool.prog.max/2
                self.tool.prog.description='loading raster'

                if (not self.currdisp is  None):
                    self.currdisp.close()
                self.currdisp = rasterio.open(in_mem)

                self.transform1 = self.currdisp.transform
                if self.tool.sampling_opts['upsampling']:
                    self.currimg = extract_upsampled_bands(self.currdisp)
                else:
                    self.currimg = self.currdisp.read().astype('float32')

                self.rgb= np.asarray(self.t2rgb.evaluate_pixel(self.currimg[3]/10000,self.currimg[2]/10000,self.currimg[1]/10000))
                vis_img=Image.fromarray(self.rgb)
                buffer=BytesIO()
                vis_img.save(buffer,'jpeg')
                dt=b64encode(buffer.getvalue())
                dt = dt.decode("ascii")
                self.image2.url="data:image/jpeg;base64,"+dt
                self.image2.bounds=((self.currdisp.bounds[1], self.currdisp.bounds[0]), (self.currdisp.bounds[3], self.currdisp.bounds[2]))
                self.tool.sampler.draw_control.polygon['pathOptions']['bounds']=self.image2.bounds
                
                #masks_saved.geo_dataframe=in_file
                #masks_saved._update_data('data')

                if (not self.tool.model.pixnn is None):
                    self.tool.model.predict_S2(self.currimg,self.currdisp)
                    self.tool.model.masks_predicted.geo_dataframe= self.tool.model.curr_pred[self.tool.model.curr_pred["Area_synthetic"] > self.tool.model_opts['cutoff_area']]
                    self.tool.model.masks_predicted._update_data('data')

            self.tool.sampler.draw_control.polygon['pathOptions']['source']=self.raster_name
            self.tool.sampler.draw_control.force=True
            self.tool.sampler.draw_control.notify_change({'name':'polygon', 'type':'change'})        
            self.tool.prog.value=self.tool.prog.max
            self.tool.prog.description='Success!'
        elif (kwargs['action']=='create' and kwargs['geo_json'][0]['properties']['type']=='polygon' and self.raster_name[0]=='n'):
            self.downloaded_rasters[self.raster_name]['counter']+=1
        elif (kwargs['action']=='remove' and kwargs['geo_json'][0]['properties']['type']=='polygon' and kwargs['geo_json'][0]['properties']['style']['source'][0]=='n'):
            name=kwargs['geo_json'][0]['properties']['style']['source']
            if (self.downloaded_rasters.__contains__(name)):
                self.downloaded_rasters[name]['counter']-=1



            
     

    def update_coordinates(self,change):
        self.centcoors[0].value=self.tool.m1.center[0]
        self.centcoors[1].value=self.tool.m1.center[1]
    def jump_to(self,b):
        self.tool.m1.center=(self.centcoors[0].value,self.centcoors[1].value)


    def vis (self,b) :
        self.image2.visible=True
        self.highresbounds.totalenergy=None
        self.highresbounds.opacity=self.highresbounds.fill_opacity=0

        if (b.description== 'Previous'):
            if (self.currind<=0): return 
            else: k=-1 
        else: k=1
        self.tool.prog.value=0
        self.tool.prog.description='loading raster'
        self.currind=(self.currind+k)%len(self.data)
        #mask_name='clustered_'+data[currind][0]+'_'+data[currind][1]+'_'+data[currind][2]+".GeoJSON"
        self.raster_name=self.data[self.currind][0]+'_'+self.data[self.currind][1]+'_'+self.data[self.currind][2]+".tiff"

        h= self.raster_name.replace(".tiff","").split("_")
        self.tool.m1.center=(float(h[1]), float(h[2]))
        if (not self.currdisp is  None):
            self.currdisp.close()
        self.currdisp = rasterio.open(f"{self.data_path}/{self.raster_name}")


        self.transform1 = self.currdisp.transform
        
        if self.tool.sampling_opts['upsampling']:
            self.currimg = extract_upsampled_bands(self.currdisp)
        else:
            self.currimg = self.currdisp.read().astype('float32')

        self.rgb= np.asarray(self.t2rgb.evaluate_pixel(self.currimg[3]/10000,self.currimg[2]/10000,self.currimg[1]/10000))

        vis_img=Image.fromarray(self.rgb)
        buffer=BytesIO()
        vis_img.save(buffer,'jpeg')
        dt=b64encode(buffer.getvalue())

        dt = dt.decode("ascii")

        self.image2.url="data:image/jpeg;base64,"+dt
        self.image2.bounds=((self.currdisp.bounds[1], self.currdisp.bounds[0]), (self.currdisp.bounds[3], self.currdisp.bounds[2]))
        self.tool.sampler.draw_control.polygon['pathOptions']['source']=self.raster_name
        self.tool.sampler.draw_control.polygon['pathOptions']['bounds']=self.image2.bounds
        self.tool.sampler.draw_control.force=True
        self.tool.sampler.draw_control.notify_change({'name':'polygon', 'type':'change'})


        #print (os.path.isfile(image2.url))
#        masks_saved.geo_dataframe=in_file
        if (not self.tool.model.pixnn is None):

            self.tool.model.predict_S2(self.currimg,self.currdisp)
            self.tool.model.masks_predicted.geo_dataframe= self.tool.model.curr_pred[self.tool.model.curr_pred["Area_synthetic"] > self.tool.model_opts['cutoff_area']]
        a = torch.cuda.memory_allocated(0)
        r= torch.cuda.memory_reserved(0)
        print(f'memory allocated:{a/(1024.0**3)}/ memory reserved:{r/(1024.0**3)}')
#        masks_saved._update_data('data')
        self.tool.model.masks_predicted._update_data('data')

        self.tool.prog.value=self.tool.prog.max
        self.tool.prog.description='Success!'
    def get_closest(self,b):
        self.image2.visible=True
        self.highresbounds.totalenergy=None
        self.highresbounds.opacity=self.highresbounds.fill_opacity=0

        self.tool.prog.value=0
        self.tool.prog.description='loading raster'
        _,indices=self.index_flat.search(np.array([list(self.tool.transformer.transform(self.tool.m1.center[0],self.tool.m1.center[1]))]),1)
        coos= self.data[indices[0,0]]
        self.raster_name=f'e_{coos[1]}_{coos[2]}.tiff'
        if (not self.currdisp is  None):
            self.currdisp.close()
        self.currdisp = rasterio.open(f"{self.data_path}/{self.raster_name}")

        self.transform1 = self.currdisp.transform
        if self.tool.sampling_opts['upsampling']:
            self.currimg = extract_upsampled_bands(self.currdisp)
        else:
            self.currimg = self.currdisp.read().astype('float32')
        self.tool.m1.center=(float(coos[1]), float(coos[2]))
        self.rgb= np.asarray(self.t2rgb.evaluate_pixel(self.currimg[3]/10000,self.currimg[2]/10000,self.currimg[1]/10000))
        vis_img=Image.fromarray(self.rgb)
        buffer=BytesIO()
        vis_img.save(buffer,'jpeg')
        dt=b64encode(buffer.getvalue())
        dt = dt.decode("ascii")
        self.image2.url="data:image/jpeg;base64,"+dt
        self.image2.bounds=((self.currdisp.bounds[1], self.currdisp.bounds[0]), (self.currdisp.bounds[3], self.currdisp.bounds[2]))
        self.tool.sampler.draw_control.polygon['pathOptions']['source']=self.raster_name
        self.tool.sampler.draw_control.polygon['pathOptions']['bounds']=self.image2.bounds
        self.tool.sampler.draw_control.force=True
        self.tool.sampler.draw_control.notify_change({'name':'polygon', 'type':'change'})
        #masks_saved.geo_dataframe=in_file
        if (not self.tool.model.pixnn is None):
            self.tool.model.predict_S2(self.currimg,self.currdisp)
            self.tool.model.masks_predicted.geo_dataframe= self.tool.model.curr_pred[self.tool.model.curr_pred["Area_synthetic"] > self.tool.model_opts['cutoff_area']]
            self.tool.model.masks_predicted._update_data('data')

        #masks_saved._update_data('data')
        self.tool.prog.value=self.tool.prog.max
        self.tool.prog.description='Success!'
    def save_predictions(self,b):
        self.tool.model.masks_predicted.geo_dataframe.to_file(f'{self.data_path}/masks/mask_{self.raster_name.replace('tiff','GeoJSON').replace("new/","")}', driver='GeoJSON',crs='EPSG:4326')
        if self.downloaded_rasters.__contains__(self.raster_name):
            with open(f'{self.tool.visualizer.data_path}/{self.raster_name}', "wb") as f:
                f.write(self.downloaded_rasters[self.raster_name]['bytes'].getbuffer()) 
            self.downloaded_rasters.pop(self.raster_name)


        



class interactive_tool :
    m1 =None
    sampler=None
    model =None
    visualizer=None
    prog=None
    model_opts={}
    sampling_opts={}
    popup=None
    transformer=None
    def __init__(self,dataset_path,config_path, data_path, tabular_data_path,device='cpu'):
        self.transformer = Transformer.from_crs("EPSG:4326","EPSG:32633")
        self.m1 =Map(close_popup_on_click=False,prefer_canvas=True,basemap=basemaps.OpenStreetMap.Mapnik,center=(0.,0.),layout=Layout(width='1900px', height='1900px'),zoom=14,scroll_wheel_zoom=True)
        self.prog=ipywidgets.IntProgress(
            value=0,
            min=0,
            max=10,
            bar_style='', # 'success', 'info', 'warning', 'danger' or ''
            style={'bar_color': 'green','description_width': 'auto'},
            orientation='horizontal'
        )
        
        pr = WidgetControl(widget=self.prog, position='bottomleft')
        self.m1.add(pr)
        self.m1.add(FullScreenControl())
        self.m1.add(self.prog)
        model(self,config_path,tabular_data_path,device)
        sample(self,dataset_path)
        visualize(self,data_path)
        self.m1.add (LayersControl())

        message1 = HTML()
        message1.value= f'Welcome!'

        self.popup = Popup(
            child=message1,
            location=(self.m1.center[0],self.m1.center[1]),
            close_button=True,
            auto_close=False,
            close_on_escape_key=False
        )
        self.m1.add(self.popup)

        
        display(self.m1)

        