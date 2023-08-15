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
           ├─ models/
           |    ├─ ..
           |    ├─ research/
           |    |    ├─ object_detection/
           |    |               ├─ ..  
           ├─ scripts/
           |    ├─ generate_tfrecord.py
           └─ workspace/
           |    ├─ annotations
           |    ├─ images
           |    ├─ models
           |    └─ pretrained-models
```
Make sure you create all these directories in your environment before begin to build the model.

## Data Acquisition
### Kaggle API

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

Once the API is copied, paste the command in the python shell, and run it.

```
!kaggle datasets download -d andrewmvd/face-mask-detection
```


