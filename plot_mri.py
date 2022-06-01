import os
import cv2
import glob
from skimage import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def plot_scan(df_pred:pd.DataFrame):
    for i in range(0, len(df_pred)):
        if df_pred['has_mask'][i] == 1:
            img = io.imread(df_pred.image_path[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            plt.axis('off')
            plt.imshow(img)
            plt.title('Original Image')
            plt.savefig('./static/predicted/image1.png', bbox_inches='tight')
            predicted_mask = np.asarray(df_pred.predicted_mask[i])[0].squeeze().round()
            plt.axis('off')
            plt.title('AI Predicted Mask')
            plt.imshow(predicted_mask)
            plt.savefig('./static/predicted/image2.png', bbox_inches='tight')
            img_ = io.imread(df_pred.image_path[i])
            img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
            img_[predicted_mask == 1] = (0, 255, 0)
            img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
            plt.axis('off')
            plt.imshow(predicted_mask)
            plt.title("MRI with AI Predicted Mask")
            plt.imshow(img_)
            plt.savefig("./static/predicted/image.png", bbox_inches='tight')
            print("Saved Predicted Image")

