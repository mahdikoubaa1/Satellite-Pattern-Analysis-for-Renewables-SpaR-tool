# Contribution (prediction on Germany S2 rasters)
## model for solarfarm segemntation (PIXNN)
### Brief description
The model can be seen as a fully connected nn having as input the bands of a single pixel bands and trained using gradient descent (GD)

We classify by colors/ by band values 


### In practice

 

we are using ADAM optimizer with a training dataset of n pixels flattened into 1 x n image 

We are using 1x1 convolution to mimic the behavior of fully connected nn and facilitate the transition to images of arbitrary size. 
 

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
* Each sample contains 13 values
* We are using all 13 bands except band 11 to train the model 
 
 
## SpaR tool
The SpaR tool is a UI interface available in notebooks/SpaR_Tool.ipynb.
# supported features
* sampling, training and prediction for pixnn architectures (S2 Rasters PV Segmentation)
* prediction for SolarSAM and Segformer (Aerial Rasters PV Segmentaion (limited to Bavaria))
## new environment requirement installation (miniconda/anaconda is required)
```bash
bash ./setup_environment.sh
```
## data link
for the SpaR tool to work as intended, please extract [data.zip](https://drive.google.com/file/d/1a3qTBY3jtcjc_LGzrPugM4NP7biVplur/view?usp=drive_link) in the root directory of this project
