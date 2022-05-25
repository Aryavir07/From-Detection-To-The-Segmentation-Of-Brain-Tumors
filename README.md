<h2 align="center"><i>Brain Tumor Detection Using Deep Learning</i></h2>
<p align="center">

  <a href="https://github.com/Aryavir07/Detecting-Brain-Tumor-Using-Deep-Learning/blob/main/LICENSE">
    <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-yellow.svg" target="_blank" />
  </a>
  <a href="https://twitter.com/aryaveersingh07">
    <img alt="Twitter: Aryaveer Singh" src="https://img.shields.io/twitter/follow/AryaveerSingh.svg?style=social" target="_blank" />
  </a>
</p>

<h2 align="center">Repository overview✔</h2>

- <a href="https://arxiv.org/abs/1904.00592" target="_blank"><strong>Using ResUNET</strong></a> and transfer learning for Brain Tumor Detection. This would lower the cost of cancer diagnostics and aid in the early detection of malignancies, which would effectively be a lifesaver. <br>To categorise MRI images including brain malignancies, this notebook provides implementations of deep learning models such as *ResNet50, VGG16 (through transfer learning), and CNN architectures*. After training on **100 epochs**, the results showed **ResNet50** and **VGG16** gave very similar results in classification. <br>This notebook uses <a href="https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation" target="_blank"><strong>Dataset</strong></a> from Kaggle containing 3930 brain MRI scans in **.tif** format along with their brain tumor location and patients information.
<br>
<u>This notebook😊 contains line by line code explanation and many Q&As 🙌</u>

## Working
- The project is based on image segmentation, and the purpose of image segmentation is to comprehend and extract information from images at the pixel level.
- Image segmentation may be used for object detection and localisation, which has a wide range of applications including medical imaging and self-driving automobiles.
- The initial portion of this project implements deep learning models such as ResNet50, two distinct architectures of the fine-tuned VGG16 model, and a rudimentary CNN model to categorise MRI scans containing brain tumor.
- In the second part, **RESUNET** model is implemented to localize brain tumor from classified MRI scans.
- Using this image segmentation neural network is trained to generate pixel-wise masks of the images.
- Modern image segmentation techniques are based on deep learning approach which makes use of common architectures such as CNN, FCNs (Fully Convolution Networks) and Deep Encoders Decoders.


### ResUNet
![ResUnet](https://user-images.githubusercontent.com/42632417/110745770-cac0be80-8261-11eb-87d3-894861b11a4c.png)

<p align="center">
  <a href="https://www.researchgate.net/figure/Illustration-of-the-proposed-Res-Unet-architecture-as-the-generator_fig2_327748708" target="_blank">Source</a> 
  and <a href="https://idiotdeveloper.com/what-is-resunet/" target="_blank">Explanation</a>
</p>

ResUNet architecture combines UNET backbone architecture with residual blocks to overcome vanishing gradient problem present in deep architecture.
ResUNet consists of **three** parts:

<ul>
  <li>Encoder : The contraction path consist of several contraction blocks, each block takes an input that passes through res-blocks followed by 2x2 max pooling. Feature maps after each block doubles, which helps the model learn complex features effectively.</li>
  <li>Decoder : In decoder each block takes in the up-sampled input from prevoius layer and concatenates with the corresponding output features from the res-block in the contraction path. this is then passed through the res-block followed by 2x2 upsampling convolution layers this helps to ensure that features learned while contracting are used while reconstructing the image.</li>
  <li>Bottleneck : The bottleneck block, serves as a connection between contraction path and expansion path.The block takes the input and then passes through
a res-block followed by 2 x 2 up-sampling convolution layers.</li>
</ul>

### Masks
<p align="center">
  <img src="https://user-images.githubusercontent.com/42632417/110747969-1e80d700-8265-11eb-9139-a7d7d6063d6b.png" height = 400 width = 450></img>
</p>
<ul>
  <li>The output produced by the image segmentation model is called MASK of the image.</li>
  <li>Mask is presented by associating pixel values with their coordinates like <strong>[[0,0],[0,0]]</strong> for black image shape and to represent this MASK we flatten it as <strong>[0,0,0,0]</strong></li>
</ul>

### Performance

| **Model Name** | **Accuracy** | **Balanced Accuracy** | **Recall** | **F1 Weighted** | **F1 Average** | **Precision** |
|:--------------:|:------------:|:---------------------:|:----------:|:---------------:|:--------------:|:-------------:|
|       CNN      |     88.61    |         87.13         |   88.616   |      88.60      |      88.61     |     88.61     |
|  VGG16 Model 1 |     82.14    |         82.68         |    82.14   |      83.39      |      82.13     |      83.14     |
|  VGG16 Model 2 |     83.03    |         81.59         |    83.03   |      83.18      |      83.03     |     83.03     |
|    ResNet50    |     98.88    |         98.71         |    98.88   |      98.88      |      98.88     |     98.88     |

### Final Results

<p align = "center">
  <img src = "https://user-images.githubusercontent.com/42632417/110748369-9fd86980-8265-11eb-8308-6639fc6fc63e.png" height = 450 width = 750 > </img>
</p>

#### Citations and Original Authors:
```
**Ryan Ahmed** [ https://www.coursera.org/instructor/~48777395 ]

@article{diakogiannis2020resunet,
  title={ResUNet-a: A deep learning framework for semantic segmentation of remotely sensed data},
  author={Diakogiannis, Foivos I and Waldner, Fran{\c{c}}ois and Caccetta, Peter and Wu, Chen},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={162},
  pages={94--114},
  year={2020},
  publisher={Elsevier}
}

@article{simonyan2014very,
  title={Very deep convolutional networks for large-scale image recognition},
  author={Simonyan, Karen and Zisserman, Andrew},
  journal={arXiv preprint arXiv:1409.1556},
  year={2014}
}
```

