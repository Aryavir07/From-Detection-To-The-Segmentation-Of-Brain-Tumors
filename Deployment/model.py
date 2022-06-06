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


class Model:

    def __init__(self):
        self.model_path = model_path
    
    @staticmethod
    def model_conv():
        """_summary_
        Returns:
            _type_: CNN model
        """
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu",
                kernel_initializer="he_uniform", padding="same", input_shape=(256, 256, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu",
                kernel_initializer="he_uniform", padding="same", input_shape=(256, 256, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu",
                kernel_initializer="he_uniform", padding="same", input_shape=(256, 256, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(units=128, activation="relu",
                kernel_initializer="he_uniform"))
        model.add(Dropout(0.5))
        model.add(Dense(units=2, activation="softmax"))
        model.compile(optimizer="adam",
                    loss="categorical_crossentropy", metrics=["accuracy"])
        print(mode.summary())
        return model
    
    @staticmethod
    def model_vgg1():
        """_summary_

        Returns:
            _type_: Fine tuned VGG 16 model 1
        """
        model = VGG16(weights="imagenet", include_top=False,
                    input_tensor=Input(shape=(224, 224, 3)))
        for layer in model.layers:
            layers.trainable = False
        head = model.output
        head = Flatten()(head)
        head = Dense(units=128, activation='relu',
                    kernel_initializer="he_uniform")(head)
        head = Dense(2, activation='softmax')(head)
        model_vgg = Model(inputs=model.input, outputs=head)
        return model_vgg

    @staticmethod
    def model_vgg2():
        model=VGG16(include_top=False,input_shape=(224,224,3))
        for layer in model.layers:
            layer.trainable=False
        head = model.output
        head = AveragePooling2D(pool_size = (4,4))(head)
        head = Flatten(name= 'flatten')(head)
        head = Dense(512, activation = "relu")(head) 
        head = Dropout(0.3)(head)
        head = Dense(256, activation = "relu")(head)
        head = Dropout(0.1)(head)
        head = Dense(128, activation = "relu")(head)
        head = Dense(2, activation = 'softmax')(head)
        model = Model(inputs = model.input, outputs = head)
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics =['accuracy'])  
        return model
    
    @staticmethod
    def model_ResNet50():
        model = ResNet50(weights = 'imagenet', include_top = False, input_tensor = Input(shape=(256, 256, 3)))
        
        for layer in model.layers:
            layers.trainable = False
        head = model.output
        head = AveragePooling2D(pool_size = (4,4))(head)
        head = Flatten(name= 'flatten')(head)
        head = Dense(256, activation = "relu")(head) 
        head = Dropout(0.3)(head)
        head = Dense(256, activation = "relu")(head)
        head = Dropout(0.3)(head)
        head = Dense(128, activation = "relu")(head)
        head = Dropout(0.3)(head)
        head = Dense(2, activation = 'softmax')(head)
        model = Model(inputs = model.input, outputs = head)
        return model
    
    @staticmethod
    def get_model():
        try:
            #Load Model
            with open('./saved_models/classifier-resnet-model.json', 'r') as json_file:
                json_Model = json_file.read()
            model = tf.keras.models.model_from_json(json_Model)
            model.load_weights('./saved_models/classifier-resnet-weights.hdf5')
            model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics= ["accuracy"])
            print("----------------------------------------------------")
            print("Classification model loaded and compiled successfully!")
            print("----------------------------------------------------")
            return model
        except Exception as err:
            print(f"Printing Error:-> {err}")