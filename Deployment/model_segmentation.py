from typing import Optional
import tensorflow as tf
from tensorflow.keras import Model  
from tensorflow import keras
from tensorflow.keras.applications import DenseNet121 # 2017 architecture
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.layers import *
from tensorflow.keras import Model 
from tensorflow.keras.models import load_model
from tensorflow import keras 
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from typing import Optional
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from loss import tversky, tversky_loss, focal_tversky


def upsample_concat(x, skip):
    x = UpSampling2D((2,2))(x)
    merge = Concatenate()([x, skip])
    return merge

def resblock(X, f):
  X_copy = X
  X = Conv2D(f, kernel_size = (1,1) ,strides = (1,1),kernel_initializer ='he_normal')(X)
  X = BatchNormalization()(X)
  X = Activation('relu')(X) 
  X = Conv2D(f, kernel_size = (3,3), strides =(1,1), padding = 'same', kernel_initializer ='he_normal')(X)
  X = BatchNormalization()(X)
  X_copy = Conv2D(f, kernel_size = (1,1), strides =(1,1), kernel_initializer ='he_normal')(X_copy)
  X_copy = BatchNormalization()(X_copy)
  X = Add()([X,X_copy])
  X = Activation('relu')(X)
  return X


class Model_Seg:
    def __init__(self, input_shape = (256,256,3), model_path = './saved_models/ResUNet-model.json'):
        self.input_shape = input_shape
        self.model_path = model_path
    
    @staticmethod
    def build_model_segment(self):
        input_shape = self.input_shape
        X_input = Input(input_shape)
        conv1_in = Conv2D(16,3,activation= 'relu', padding = 'same', kernel_initializer ='he_normal')(X_input)
        conv1_in = BatchNormalization()(conv1_in)
        conv1_in = Conv2D(16,3,activation= 'relu', padding = 'same', kernel_initializer ='he_normal')(conv1_in)
        conv1_in = BatchNormalization()(conv1_in)
        pool_1 = MaxPool2D(pool_size = (2,2))(conv1_in)
        conv2_in = resblock(pool_1, 32)
        pool_2 = MaxPool2D(pool_size = (2,2))(conv2_in)
        conv3_in = resblock(pool_2, 64)
        pool_3 = MaxPool2D(pool_size = (2,2))(conv3_in)
        conv4_in = resblock(pool_3, 128)
        pool_4 = MaxPool2D(pool_size = (2,2))(conv4_in)
        conv5_in = resblock(pool_4, 256)
        up_1 = upsample_concat(conv5_in, conv4_in)
        up_1 = resblock(up_1, 128)
        up_2 = upsample_concat(up_1, conv3_in)
        up_2 = resblock(up_2, 64)
        up_3 = upsample_concat(up_2, conv2_in)
        up_3 = resblock(up_3, 32)
        up_4 = upsample_concat(up_3, conv1_in)
        up_4 = resblock(up_4, 16)
        output = Conv2D(1, (1,1), padding = "same", activation = "sigmoid")(up_4)
        model_seg = Model(inputs = X_input, outputs = output)
        adam = tf.keras.optimizers.Adam(learning_rate = 0.05, epsilon = 0.1)
        model_seg.compile(optimizer = adam, loss = focal_tversky, metrics = [tversky])
        return model_seg
    
    @staticmethod
    def get_model():
        try:
            with open('./saved_models/ResUNet-model.json', 'r') as json_file:
                json_savedModel= json_file.read()
            model_seg = tf.keras.models.model_from_json(json_savedModel)
            model_seg.load_weights('./saved_models/ResUNet-weights.hdf5')
            adam = tf.keras.optimizers.Adam(learning_rate = 0.05, epsilon = 0.1)
            model_seg.compile(optimizer = adam, loss = focal_tversky, metrics = [tversky])
            print("----------------------------------------------------")
            print("Segmentation model loaded and compiled successfully!")
            print("----------------------------------------------------\n")
            return model_seg
        except Exception as err:
            print(f"Printing Error:-> {err}")