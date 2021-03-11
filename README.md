# Detecting-Brain-Tumor-Using-Deep-Learning
## Project Overview
- Goal of this project is to detect and localize brain tumors based on MRI scans using deep learning model.
- This would reduced reduce the cost of cancer diagnosis and help in early diagnosis of tumors which would essentially be a life saver. 
- Dataset contains 3930 Brain MRI scans in .tif format along with their brain tumor location.

## How?
- Project is based on Image segmentation
- The goal of image segmentation is to understand and extract information from images at the pixel-level.
- Image Segmentation can be used for object recognition and localization which offers tremendous value in many applications such as medical imaging and self-driving cars etc.
- Using image segmentation a neural network will be trained to produced pixel-wise mask of the image.
- Modern image segmentation techniques are based on deep learning approach which makes use of common architectures such as CNN, FCNs (Fully Convolution Networks) and Deep Encoders Decoders.
- Here in this project, Res-U-Net architecture is used to perform this task
