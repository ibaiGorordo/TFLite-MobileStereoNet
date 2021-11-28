# TFLite-MobileStereoNet
Python scripts for performing stereo depth estimation using the MobileStereoNet model in Tensorflow Lite.

![MobileStereoNet depth estimation TFLite](https://github.com/ibaiGorordo/TFLite-MobileStereoNet/blob/main/doc/out.jpg)
*Stereo depth estimation on the cones images from the Middlebury dataset (https://vision.middlebury.edu/stereo/data/scenes2003/)*

# Requirements

 * Check the **requirements.txt** file. Additionally, **pafy** and **youtube-dl** are required for youtube video inference.
 * DrivingStereo dataset, **ONLY** for the `driving_sereo_test.py`script. Link: https://drivingstereo-dataset.github.io/
 
# Installation
```
pip install -r requirements.txt
pip install pafy youtube-dl
```

# TFLite model
The original models was converted to different formats (including .tflite) by [PINTO0309](https://github.com/PINTO0309), the models can be found in [his repository](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/150_MobileStereoNet).

# Original Pytorch model
The Pytorch pretrained model was taken from the [original repository](https://github.com/cogsys-tuebingen/mobilestereonet).
 
# Examples

 * **Image inference**:
 
 ```
 python image_depth_estimation.py 
 ```
 
  * **Video inference**:
 
 ```
 python video_depth_estimation.py
 ```
 
 * **DrivingStereo dataset inference**:
 
 ```
 python driving_sereo_test.py
 ```
 
# [Inference video Example](https://youtu.be/6aQ8nqbabs8) 
 ![MobileStereoNet depth estimation ONNX](https://github.com/ibaiGorordo/TFLite-MobileStereoNet/blob/main/doc/video_stereo_depth.gif)

# References:
* MobileStereoNet model: https://github.com/cogsys-tuebingen/mobilestereonet
* PINTO0309's model zoo: https://github.com/PINTO0309/PINTO_model_zoo
* PINTO0309's model conversion tool: https://github.com/PINTO0309/openvino2tensorflow
* DrivingStereo dataset: https://drivingstereo-dataset.github.io/
* Original paper: https://arxiv.org/pdf/2108.09770.pdf
 
