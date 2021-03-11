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
- Here in this project, Res-U-Net architecture is used to perform this task.

# ResUNet
![ResUnet](https://user-images.githubusercontent.com/42632417/110745770-cac0be80-8261-11eb-87d3-894861b11a4c.png)
Source: https://www.researchgate.net/figure/Illustration-of-the-proposed-Res-Unet-architecture-as-the-generator_fig2_327748708
Explanation: https://idiotdeveloper.com/what-is-resunet/
- ResUNet architecture combines UNET backbone architecture with residual blocks to overcome vanishing gradient problem present in deep architecture.
- ResUNet consists of three parts:
1. Encoder or contracting Path
2. Bottleneck
3. Decoder or expansive path

- Encoder : The contraction path consist of several contraction blocks, each block takes an input that passes through res-blocks followed by 2x2 max pooling. Feature maps after each block doubles, which helps the model learn complex features effectively.
- Decoder : In decoder each block takes in the up-sampled input from prevoius layer and concatenates with the corresponding output features from the res-block in the contraction path. this is then passed through the res-block followed by 2x2 upsampling convolution layers.
- this helps to ensure that features learned while contracting are used while reconstructing the image.
- Bottleneck : The bottleneck block, serves as a connection between contraction path and expansion path.The block takes the input and then passes through
a res-block followed by 2 x 2 up-sampling convolution layers.
