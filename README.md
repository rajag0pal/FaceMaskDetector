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
If protoc is not installed in your machine, we need to download it explictly by heading towards 'protoc releases' github page (https://github.com/protocolbuffers/protobuf/releases)

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

If you have a GPU hardware component in your local machine, make sure that you have the one that has **CUDA** support. CUDA stands for Compute Unified Device Architecture, which is a parallel computing platform and API model developed by Nvidia. <br>
To know the list of CUDA enabled NVIDIA GPUs, check this link. https://developer.nvidia.com/cuda-gpus

![image](https://github.com/rajag0pal/FaceMaskDetector/assets/80576855/c084b2db-c32d-4680-b95b-765aae374199)

### Tensorflow and CUDA

**Note** : Tensorflow versions influence directly with the CUDA version. Installing incompatible versions doesn't make tensorflow to detect GPUs.

![image](https://github.com/rajag0pal/FaceMaskDetector/assets/80576855/04e96883-feb4-4f8f-b85e-ef8b9191ff76)

### cuDNN 

The NVIDIA CUDA® Deep Neural Network library (cuDNN) is a GPU-accelerated library of primitives for deep neural networks. cuDNN provides highly tuned implementations for standard routines such as forward and backward convolution, pooling, normalization, and activation layers.
- Tensor Core acceleration for all popular convolutions including 2D, 3D, Grouped, Depth-wise separable, and Dilated with NHWC and NCHW inputs and outputs
- Optimized kernels for computer vision and speech models including ResNet, ResNext, EfficientNet, EfficientDet, SSD, MaskRCNN, Unet, VNet, BERT, GPT-2, Tacotron2 and WaveGlow
- Support for FP32, FP16, BF16 and TF32 floating point formats and INT8, and UINT8 integer formats
- Support for fusion of memory-limited operations like pointwise and reduction with math-limited operations like convolution and matmul
- Support for Windows and Linux with the latest NVIDIA data center and mobile GPUs.

I have **NVIDIA GeForce GTX 1650** GPU with 4GB memory. This supprots CUDA. And I installed this combination of Tensorflow 2.6 | CUDA 11.2 | cuDNN 8.1

- CUDA 11.2 ToolKit - https://developer.nvidia.com/cuda-11.2.0-download-archive?target_os=Windows&target_arch=x86_64
- cuDNN 8.1 - https://developer.nvidia.com/cudnn

Once the files are downloaded, the installation process is simple and automative. All you need to do is to choose the installation directory. The CUDA installer will automatically install the required drivers. 

![image](https://github.com/rajag0pal/FaceMaskDetector/assets/80576855/542c0191-88f0-4268-9338-6df53eba7b98)

After this, go to the installed directory. Usually the default is User/Program Files/NVIDIA GPU Computing Toolkit/CUDA

![image](https://github.com/rajag0pal/FaceMaskDetector/assets/80576855/2e59fb65-fb9d-48ca-ac79-5192aaab4dda)

cuDNN is a zipped file.

![image](https://github.com/rajag0pal/FaceMaskDetector/assets/80576855/cafa84a6-bdb7-4a70-ac83-27cbb591e74b)

```
cuDNN
  ├─ bin
  ├─ include
  └─ lib
```
Copy all the files from these directories and paste those in the same named corresponding directories in User/Program Files/ NVIDIA GPU Computing Toolkit/CUDA. Once we done with this, add these directory PATH to the Environment Variable. 

## GPU check

We can check after the CUDA installation, that whether the tensorflow API we have by now, does have an access to our GPU or not, through a code snippet.<br>
Go to your env or cmd
```
>> C:\Users >python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
Output:
```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```
If tensorflow doesn't got the access, the output will be an empty list, when means CUDA is not properly configured.
```
[]
```

Alternatively, we can also check the GPU installation, using this command.

Go to your cmd
```
C:\Users > nvidia-smi
```
Output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.89       Driver Version: 460.89       CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  GeForce GTX 1650   WDDM  | 00000000:01:00.0 Off |                  N/A |
| N/A   42C    P8     4W /  N/A |    204MiB /  4096MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```
We will get exactly like this, If CUDA was installed properly.

## Inference

![image](https://github.com/rajag0pal/FaceMaskDetector/assets/80576855/49f37a82-b87d-4373-8919-7a06f29f5d22)

The above results are from the custom trained ssd_mobilenet_v2_fpnlite_320x320_coco17 model. The results are better than EfficientDET this time, but again still require some improvements.

- Result 1 : The bounding box provides a perfect detection on person not wearing mask
- Result 2 : Unlike EfficientDET, MobileNET SSD behaves way better when someone covers the mouth with shirt.
- Result 3 : The same issue with MobileNET SSD as well, that even covering mouth with hands detected as 'with_mask'
- Result 4 : Detected perfectly
- Result 5 : Here the model needs more fine tuning. It detects correctly, but the mask is not weared on mouth.
- Result 6 : Result is good, though mask is still in the picture, it is not weared, hence No mask.
- Result 7 : This time also the result is good. It says 91% a mask_weared_incorrect.
