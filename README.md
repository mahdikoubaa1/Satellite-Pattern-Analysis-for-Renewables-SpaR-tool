#  Sentinel-2 (S2) and Aerial PV Segmentation
## model for S2 raster solarfarm segemntation (PIXNN)
### Brief description
The model can be seen as a fully connected nn having as input the bands of a single pixel bands and trained using gradient descent (GD)

We classify by colors/ by band values 


### In practice

* we are using ADAM optimizer with a training dataset of n pixels flattened into 1 x n image

* We are using 1x1 convolution to mimic the behavior of fully connected nn and facilitate the transition to images of arbitrary size. 
 

### Model Architecture

 
 

* Conv2d(input_bands, 512, kernel_size=1)

* BatchNorm2d(512)

* ReLU()
 

* Conv2d(512, 512, kernel_size=1)

* BatchNorm2d(512)

* ReLU()

 

* Conv2d(512, 16, kernel_size=1)

* BatchNorm2d(16)

* ReLU()



* Conv2d(16, 2, kernel_size=1)

* BatchNorm2d(2)
 
* SoftMax 

 
 

### Training Loss
Focal Loss 

 
 

### Sampling

* The training dataset is a set of pixels that we sample from labeled polygons drawn on the raster. 
* Pixels sampled from a polygon get the same label (solarfarm or not solarfarm) as the polygon. 
* Each sample contains 13 values.
* For the best model variant we are using all 13 bands except band 11 to train the model.

## Aerial Segmentation models
### Segformer architecture using mit-b5 encoder finetuned on multi-scale PV dataset ([link](https://huggingface.co/docs/transformers/en/model_doc/segformer)).
### SolarSam architecture using SAM encoder trained on bavarian PV dataset ([link](https://ieeexplore.ieee.org/document/10738071)).
## SpaR tool
The SpaR tool is a UI interface available in notebooks/SpaR_Tool.ipynb.
### Supported Features
* sampling, training and prediction for PIXNN architecture (S2 Rasters PV Segmentation).
* prediction for SolarSAM and Segformer architectures (Aerial Rasters PV Segmentaion (currently limited to Bavaria)).
## Environment requirement installation (miniconda/anaconda is required)
```bash
bash ./setup_environment.sh
```
## Data Link
For the SpaR tool to work as intended, please extract [data.zip](https://drive.google.com/file/d/1a3qTBY3jtcjc_LGzrPugM4NP7biVplur/view?usp=drive_link) in the root directory of this project
