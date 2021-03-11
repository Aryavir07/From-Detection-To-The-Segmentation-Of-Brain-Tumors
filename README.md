# Detecting-Brain-Tumor-Using-Deep-Learning
## Project Overview
- Goal of this project is to detect and localize brain tumors based on MRI scans using deep learning model.
- This would reduce the cost of cancer diagnosis and help in early diagnosis of tumors which would essentially be a life saver. 
- Dataset contains 3930 Brain MRI scans in .tif format along with their brain tumor location.

**Dataset** : https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation

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

Other resources : https://arxiv.org/abs/1505.04597 and https://arxiv.org/abs/1904.00592

### What is Mask?
- The output produced by the image segmentation model is called MASK of the image.
- Mask is presented by associating pixel values with their coordinates like [[0,0],[0,0]] for black image shape.
- to represent this MASK we flatten it as [0,0,0,0].
*Visualization*
![download](https://user-images.githubusercontent.com/42632417/110747969-1e80d700-8265-11eb-9139-a7d7d6063d6b.png)


## Deep Learning Pipeline
![Capture](https://user-images.githubusercontent.com/42632417/110747432-62bfa780-8264-11eb-9a7e-ed64ad0ece4e.GIF)
Image credit : Ryan Ahmed [ https://www.coursera.org/instructor/~48777395 ]

# Final results
![download](https://user-images.githubusercontent.com/42632417/110748369-9fd86980-8265-11eb-8308-6639fc6fc63e.png)
