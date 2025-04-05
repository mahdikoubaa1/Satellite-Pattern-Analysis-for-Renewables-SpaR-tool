import rasterio
import numpy as np
import os
from tqdm import tqdm
from rgb_js import rgbAdjustment
import argparse

class Tiff2Rgb:
    def __init__(self, input_file, output_dir):
        self.input_file = input_file
        self.output_dir = output_dir

    def extract(self):
        with rasterio.open(self.input_file) as src0:
            band = src0.read()
        bands_np = np.array(band)
        return bands_np[3], bands_np[2], bands_np[1] # RGB
    
    def save(self, img, prefix=''):
        os.makedirs(self.output_dir, exist_ok=True)
        filename = self.input_file.split('/')[-1].split('.')[0:-1]
        filename = '.'.join(filename)
        img.save(f'{self.output_dir}/{prefix}_{filename}'+'.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data/tiff')
    parser.add_argument('--output_dir', type=str, default='data/png')
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    for tif in tqdm(os.listdir(input_dir), desc='Processing'):
        if tif.endswith('.tiff') or tif.endswith('.tif'):
            tiff2rgb = Tiff2Rgb(f'{input_dir}/{tif}', output_dir)
            bands = tiff2rgb.extract()
            img = rgbAdjustment().evaluate_pixel(bands[0]/10000, bands[1]/10000, bands[2]/10000)
            tiff2rgb.save(img,'n')