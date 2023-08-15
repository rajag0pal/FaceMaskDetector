# FaceMaskDetector
![Static Badge](https://img.shields.io/badge/google-colab-%23F9AB00?logo=googlecolab)
![Static Badge](https://img.shields.io/badge/conda-grey?logo=anaconda)




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



