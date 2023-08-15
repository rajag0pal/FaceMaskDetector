# FaceMaskDetector
![Static Badge](https://img.shields.io/badge/Python-3.7-grey?logo=python)
![Static Badge](https://img.shields.io/badge/google-colab-%23F9AB00?logo=googlecolab)
![Static Badge](https://img.shields.io/badge/conda-grey?logo=anaconda)
![Static Badge](https://img.shields.io/badge/Tensorflow-1.x-grey?logo=tensorflow)
![Static Badge](https://img.shields.io/badge/Tensorflow-2.6-grey?logo=tensorflow)
![Static Badge](https://img.shields.io/badge/cuda-11.2-grey?logo=nvidia)
![Static Badge](https://img.shields.io/badge/cDNN-8.1-grey?logo=nvidia)






### Custom Object detection model for detecting face mask wear in real time using Tensorflow Object Detection API.

```
                              Object Detection
                                      |
                                      V
                             Face mask detection
                                    /  \
                                   /    \
                      EfficientDET       MobileNET SSD   
```

Leveraged pretrained models like **Mobilenet SSD** and **EfficientDET** from Tensorflow object detection API, to train a custom face detection model. Using kaggle dataset API, face mask dataset was acquired and retrained from the pretrained checkpoints.  

## File Structure

```
FaceMaskDetector/
        ├─ Tensorflow/
        |  ├─ models/
        |  |    ├─ ..
        |  |    ├─ research/
        |  |    |    ├─ object_detection/
        |  |    |               ├─ ..  
        |  ├─ scripts/
        |  |    ├─ generate_tfrecord.py
        |  └─ workspace/
        |  |    ├─ annotations
        |  |    ├─ images
        |  |    |   ├─ allimages
        |  |    |   ├─ train
        |  |    |   └─ test
        |  |    ├─ models
        |  |    └─ pretrained-models
```
Make sure you create all these directories in your environment before begin to build the model.

## Data Acquisition
### ![Static Badge](https://img.shields.io/badge/kaggle-grey?logo=kaggle) Kaggle API

Navigate to your kaggle account, and in settings, find something called **API**. And then generate new token. A json (kaggle.json) file will be dowloaded.

![image](https://github.com/rajag0pal/FaceMaskDetector/assets/80576855/530378bc-62aa-4856-9c80-af4692e85247)

```
!pip install kaggle
```

Once installing kaggle API, bring the json file downloaded in the previous stage to the environment and run the following command.

```
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```
Now head back to the dataset repo, and copy the API command.

![image](https://github.com/rajag0pal/FaceMaskDetector/assets/80576855/008861a6-01c4-43bb-8a9e-fb95f573c98d)

Once the API command is copied, paste it in the python shell, and run it.

```
!kaggle datasets download -d andrewmvd/face-mask-detection
```

Once the dataset has been downloaded using kaggle API, it has to be unzipped accordingly to the respective directories. The face mask dataset zip file contains the following files.

```
face-mask-detection.zip
        ├─ images
        └─ annotations
```

Dataset has to be unzipped to 'Tensorflow/workspaces/allimages' directory and then train test split has to be done.

```
# unzipping images seperately
!unzip -j face-mask-detection.zip 'images/*' -d Tensorflow/workspace/images/allimages

# unzipping annoations seperately
!unzip -j face-mask-detection.zip 'annotations/*' -d Tensorflow/workspace/images/allimages
```

### Train Test Split

Use the following scrpit to seperate data into train and test

```
import os
import random
import shutil

# Set the paths for your source and destination folders
source_folder = 'allimages/'
train_folder = 'train/'
test_folder = 'test/'

# Set the desired split ratio
split_ratio = 0.8

# Get a list of all image files in the source folder
image_files = [f for f in os.listdir(source_folder) if f.endswith('.png')]

# Shuffle the list of image files
random.shuffle(image_files)

# Calculate the split point
split_index = int(len(image_files) * split_ratio)

# Split the image files
train_images = image_files[:split_index]
test_images = image_files[split_index:]
```
```
# Move images and annotations to train folder
for image in train_images:
    annotation = image.replace('.png', '.xml')
    shutil.move(os.path.join(source_folder, image), os.path.join(train_folder))
    shutil.move(os.path.join(source_folder, annotation), os.path.join(train_folder))
```
```
# Move images and annotations to test folder
for image in test_images:
    annotation = image.replace('.png', '.xml')
    shutil.move(os.path.join(source_folder, image), os.path.join(test_folder))
    shutil.move(os.path.join(source_folder, annotation), os.path.join(test_folder))
```

## ![Static Badge](https://img.shields.io/badge/Object_Detection_API-grey?logo=tensorflow) Tensorflow Object Detection API

clone the API from this github repo https://github.com/tensorflow/models

```
!cd Tensorflow && git clone https://github.com/tensorflow/models
```

Change the current directory to models/research/

```
cd Tensorflow/models/research
```

### Protocol Buffers

```
!protoc object_detection/protos/*.proto --python_out=.
```
If protoc is not installl in your machine, we need to download it explictly by heading towards protoc releases github page (https://github.com/protocolbuffers/protobuf/releases)

Go to Assets -> select the version for the os you have

Protoc 24.0-win32 - https://github.com/protocolbuffers/protobuf/releases/download/v24.0/protoc-24.0-win32.zip <br>
Protoc 24.0-win64 - https://github.com/protocolbuffers/protobuf/releases/download/v24.0/protoc-24.0-win64.zip
### Installing the Dependencies
```
!cp object_detection/packages/tf2/setup.py .
!python -m pip install .
```

### For detailed Tensorflow Object Detection API tutorial
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html

# ![Static Badge](https://img.shields.io/badge/nvidia-grey?logo=nvidia) Local NVIDIA GPU Setup 
