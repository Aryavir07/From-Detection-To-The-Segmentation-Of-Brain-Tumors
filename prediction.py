import cv2
import numpy as np
from skimage import io
from keras_preprocessing.image import ImageDataGenerator

class Prediction:
  def __init__(self, test, model, model_seg):
    self.test = test
    self.model = model
    self.model_seg = model_seg
   
  def make_prediction(self):
    test = self.test
    model = self.model
    model_seg = self.model_seg
    for i in test:
      path = str(i)
      img = io.imread(path)
      img = img * 1./255.
      img = cv2.resize(img,(256,256))
      img = np.array(img, dtype = np.float64)
      img = np.reshape(img, (1,256,256,3))
      is_defect = model.predict(img)

      if np.argmax(is_defect) == 0:
        print("Hurray! No tumor detected")
        return [i, 'No mask', 0]
      
      img = io.imread(path)
      X = np.empty((1, 256, 256, 3))
      img = cv2.resize(img,(256,256))
      img = np.array(img, dtype = np.float64)
      img -= img.mean()
      img /= img.std()
      X[0,] = img

      predict = model_seg.predict(X)

      if predict.round().astype(int).sum() == 0:
          return [i, 'No mask', 0]
      else:
        print("----------------------------------------------------")
        print("Oops! Tumor detected")
        print("----------------------------------------------------")
        print("Getting tumor location..")
        print("----------------------------------------------------")
        return [i, predict, 1]
  
  
  
  
  