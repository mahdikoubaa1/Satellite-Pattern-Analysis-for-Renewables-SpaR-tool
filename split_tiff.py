from PIL import Image
import numpy as np
import os
from pathlib import Path
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
import json
import pycrs

from tqdm import tqdm
import geopandas as gpd
#only tested with 2500x2500 images
class Split:
    def __init__(self, in_size = 1200, dest_size = 256, artificial_overlap = 0):
        self.dest_size_ = dest_size
        self.in_size_ = in_size
        if artificial_overlap < 1 and artificial_overlap >= 0:
            self.artificial_overlap_ = artificial_overlap
        else:
            raise ValueError('Artificial overlap must be between 0 and 0.99 (1 would mean 100pct overlap)')
        self.split_coos_ = self.calcSplit(self.in_size_)

    def calcZoom(self,tiff_coos,crs):
        x_min,y_min, x_max, y_max=(3*tiff_coos[0]+tiff_coos[2])/4 , (3*tiff_coos[1]+tiff_coos[3])/4, (tiff_coos[0]+3*tiff_coos[2])/4, (tiff_coos[1]+3*tiff_coos[3])/4
        bbox = box(x_min,y_min ,x_max ,y_max )
        geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=crs)

        
        return [json.loads(geo.to_json())['features'][0]['geometry']] , (x_min,y_min,x_max,y_max)
    def calcSplit(self, width_height):
        #only one calculation because working with squared formats
        #calculating the overlap on each side (half_overlap)
        #calculating an artificial overlap to based on input 0-0.99 to create more trainings images
        px_artificial_overlap = int(self.dest_size_*self.artificial_overlap_)
        art_left = self.dest_size_-px_artificial_overlap
        t = width_height//art_left
        left = width_height%art_left
        total_overlap = art_left-left
        overlap = (total_overlap//t)+px_artificial_overlap

        split_coos = []

        for r in range(t):
            for c in range(t):
                #xmin,ymin,xmax,ymax
                split_coos.append((c*self.dest_size_-c*overlap,r*self.dest_size_-r*overlap,(c+1)*self.dest_size_-c*overlap,(r+1)*self.dest_size_-r*overlap))
            #to make sure that the whole image is covered (columns)    
            split_coos.append((width_height-self.dest_size_,r*self.dest_size_-r*overlap,width_height,(r+1)*self.dest_size_-r*overlap))
        #to make sure that the whole image is covered (rows)
        for c in range(t):
                split_coos.append((c*self.dest_size_-c*overlap,width_height-self.dest_size_,(c+1)*self.dest_size_-c*overlap,width_height))
        #bottom right corner
        split_coos.append((width_height-self.dest_size_,width_height-self.dest_size_,width_height,width_height))
        
        return split_coos
    

    def splitImages(self, image_filepath,onlyfiles):
        images = []
        geo_infos = {}
        for filename in tqdm(onlyfiles, desc="Splitting images"):
            with rasterio.open(f"{image_filepath}/{filename}") as src:
                band = src.read()
                image = np.array(band)
            tiff = rasterio.open(f"{image_filepath}/{filename}")
            tiff_coos = tiff.bounds #xmin=0,ymin=1,xmax=2,ymax=3 (utm32)
            q = 0
            for coos in self.split_coos_:
                #print(coos, image.shape())
                image_name = filename.split('.')[0:-1]
                image_name = '.'.join(image_name)
                split_image = image[0:13, coos[0]:coos[2], coos[1]:coos[3]] #coos = (xstart,ystart,xend,yend)
                images.append((split_image,image_name,q))
                geo_infos.update(self.splitImgGeoInfo(f'{image_name}_{q}', tiff_coos, coos))
                q += 1
        
        return images, geo_infos    
    
    def zoomImages(self, image_filepath,onlyfiles):
        images = []
        geo_infos = {}
        for filename in tqdm(onlyfiles, desc="Zooming images"):
            tiff = rasterio.open(f"{image_filepath}/{filename}")
            tiff_coos = tiff.bounds #xmin=0,ymin=1,xmax=2,ymax=3 (utm32)
            coos, x = self.calcZoom(tiff_coos,crs=tiff.crs.data)
            out_img, out_transform = mask(dataset=tiff, shapes=coos, crop=True)
            out_meta = tiff.meta.copy()
            epsg_code = int(tiff.crs.data['init'][5:])
            out_meta.update({"driver": "GTiff",
                 "height": out_img.shape[1],
                 "width": out_img.shape[2],
                 "transform": out_transform,
                 "crs": pycrs.parse.from_epsg_code(epsg_code).to_proj4()}
                         )
            image_name = 'zoomed_'+ f"{filename}"
            images.append((out_img,image_name, 'zoomed'))
            geo_infos.update({image_name: (x[0],x[1] ,x[2] ,x[3] , out_img.shape[1])})
            print (geo_infos)
            #print(coos, image.shape())
        
        return images, geo_infos
    def zoomImages_images_only(self, image_filepath,onlyfiles):
        images = []
        for filename in tqdm(onlyfiles, desc="Zooming images"):
            tiff = rasterio.open(f"{image_filepath}/{filename}")
            tiff_coos = tiff.bounds #xmin=0,ymin=1,xmax=2,ymax=3 (utm32)
            coos, x = self.calcZoom(tiff_coos,crs=tiff.crs.data)
            out_img, out_transform = mask(dataset=tiff, shapes=coos, crop=True)
            out_meta = tiff.meta.copy()
            epsg_code = int(tiff.crs.data['init'][5:])
            out_meta.update({"driver": "GTiff",
                 "height": out_img.shape[1],
                 "width": out_img.shape[2],
                 "transform": out_transform,
                 "crs": pycrs.parse.from_epsg_code(epsg_code).to_proj4()}
                         )
            images.append(out_img.astype('float32').reshape((13,-1)))
            #print(coos, image.shape())
        
        return images
    
    def ExtractImages(self, image_filepath,onlyfiles):
        images = []
        geo_infos = {}
        for filename in tqdm(onlyfiles, desc="Extracting images"):
            tiff = rasterio.open(f"{image_filepath}/{filename}")
            tiff_coos = tiff.bounds #xmin=0,ymin=1,xmax=2,ymax=3 (utm32)
            band = tiff.read()
            image = np.array(band)
            image_name = 'clipped_'+ f"{filename}"
            images.append((image,image_name, 'clipped'))
            geo_infos.update({image_name: (tiff_coos[0],tiff_coos[1],tiff_coos[2],tiff_coos[3], image.shape[1])})
            #print(coos, image.shape())
        return images, geo_infos

    def ExtractImages_imagesonly(self, image_filepath,onlyfiles):
        images = []
        geo_infos = {}
        for filename in tqdm(onlyfiles, desc="Extracting images"):
            tiff = rasterio.open(f"{image_filepath}/{filename}")
            tiff_coos = tiff.bounds #xmin=0,ymin=1,xmax=2,ymax=3 (utm32)
            band = tiff.read()
            image = np.array(band)
            images.append(image.astype('float32').reshape((13,-1)))
            #print(coos, image.shape())
        return images
    def splitMask(self, mask_filepath):
        masks = []
        for filename in os.listdir(mask_filepath):
            mask = np.load(f"{mask_filepath}/{filename}")
            q = 0
            for coos in self.split_coos_:
                mask_name = filename.split('.')[0]
                new_mask = mask[coos[1]:coos[3],coos[0]:coos[2]]
                masks.append((new_mask,mask_name,q))
                q += 1
        
        return masks

    

    def splitBoundingBox(self, bounding_box, image_id):

        new_bounding_box = {}
        q = 0
        for coos in self.split_coos_:
            #possibility that box between new image tiles --> only in the image which contains the whole box could be changed?
            if bounding_box[0] > coos[0] and bounding_box[1] > coos[1] and bounding_box[2] < coos[2] and bounding_box[3] < coos[3]:
                  new_bounding_box[f"{image_id}_{q}"] = (bounding_box[0]-coos[0],bounding_box[1]-coos[1],bounding_box[2]-coos[0],bounding_box[3]-coos[1])
            q += 1
        return new_bounding_box
    
    def splitImgGeoInfo(self, image_name, tiff_coos, coos):
        
        x_min = (coos[0]/self.in_size_) * (tiff_coos[2]-tiff_coos[0]) + tiff_coos[0]
        y_min = -(coos[1]/self.in_size_) * (tiff_coos[3]-tiff_coos[1]) + tiff_coos[3]
        x_max = (coos[2]/self.in_size_) * (tiff_coos[2]-tiff_coos[0]) + tiff_coos[0]
        y_max = -(coos[3]/self.in_size_) * (tiff_coos[3]-tiff_coos[1]) + tiff_coos[3]

        split_img_coos = {image_name: (x_min, y_min, x_max, y_max, self.dest_size_)}

        return split_img_coos