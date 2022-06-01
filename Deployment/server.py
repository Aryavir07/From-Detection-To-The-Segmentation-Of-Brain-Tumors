import numpy as np
import pandas as pd
from model import Model
from model_segmentation import Model_Seg
from prediction import Prediction
from plot_mri import plot_scan

def get_prediction():
        model = Model.get_model()
        model_seg = Model_Seg.get_model()
        path = ["./test_image.tif"]
        obj = Prediction(path, model, model_seg)
        result = obj.make_prediction()
        df = pd.DataFrame([result])
        df.columns = ["image_path", "predicted_mask", "has_mask"]
        plot_scan(df)
        
get_prediction()

