# blurPoseDetect
This repo implements a MobileNetV2 based neural network for prediction of human face pose (euler angle) and image blurriness.
## Overview:
- **fast and multitask MobileNetv2**： We modified the MobileNetv2 model structure to perform dual tasks of face blur detection and Euler angle pose prediction. The input image is compressed into an RGB image of 112x112 pixels, and width-multi is set to 0.25 to speed up the inference.
- **blur detection**：Face blurriness score, output 1 of the model. It detects motion blur, Gaussian blur, and defocus blur, and is robust to image noises.

- **pose detection**：The euler angle (yaw, pitch, roll) of human faces, output 2 of the model. It is robust to noises and slight blur interference.


## Prerequisites
- pyTorch>=1.1.0.
- ncnn：for model conversion
- [onnx simplifier](https://github.com/daquexian/onnx-simplifier)：for model deployment
### Optional
- hdface：For dataset preparation, e.g. cropping faces from images.

## Code structure

- **Train**：`train.py`

We train the model on multiple datasets. 

The dataset, dataloader, image preprocessing, and data augmentation are defined in `utils.py`.
- **Test**：`test.py`
Test on images downloaded from internet, which are not used for training and validation.

- **Model architecture**: `mobileNetv2.py`

Modified based on [mobileNetv2](https://github.com/tonylins/pytorch-mobilenet-v2), adding dual task functionality.

- **Model conversion**: `toNcnn.py`

`Pipeline: pyTorch model -> onnx model -> simplified onnx model -> ncnn model`

The ncnn model can be validated in `ncnnTest.cpp`.


## Some methods of data augmentation

- **Blur detection:**
  Add random blurriness to clear images while loading data, and calibrate the blurriness score ground truth after data augmentation.
   对较清晰图片，在dataloader处随机加上模糊，并计算模糊后的blur值。
- **Pose estimation:**
  - Add random rotation to images with small yaw whiling loading data, and calculate the roll angle value after the rotation (when yaw is small, rotating the image has negligible effect on yaw and pitch).
  -  Rotate images offline in the 300WLP and Biwi Kinect datasets and annotate the Euler angles with PRNet. The disadvantage of this method is that PRNet computes Euler angles very slowly and offline storage of images is required.

Code for data augmentation code can be found in `utils.FaceDataset`.
   
