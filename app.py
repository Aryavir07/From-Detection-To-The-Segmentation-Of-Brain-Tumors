from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import secrets
import glob
import numpy as np
import pandas as pd
from model import Model
from model_segmentation import Model_Seg
from prediction import Prediction
from plot_mri import plot_scan

secret_key = secrets.token_hex(16)

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = secret_key
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'tif'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# @app.route('/predict/',methods=['GET','POST'])
def get_prediction(filename):
        model = Model.get_model()
        model_seg = Model_Seg.get_model()
        path = [f"./static/uploads/{filename}"]
        obj = Prediction(path, model, model_seg)
        result = obj.make_prediction()
        df = pd.DataFrame([result])
        df.columns = ["image_path", "predicted_mask", "has_mask"]
        if(df.shape[0] == 0):
            return False
        plot_scan(df)
        return True
    
 
@app.route('/')
def home():
    files = glob.glob("static/uploads/*")
    for f in files:
        os.remove(f)
    files = glob.glob("static/predicted/*")
    for f in files:
        os.remove(f)
    return render_template('index.html')
 
@app.route('/predict/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
        
    file = request.files['file']
    if file.filename == '':
        flash('No image selected')
        return redirect(request.url)
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        result = get_prediction(filename)
        if(result):
            return render_template('predict.html', filename=filename)
        else:
            flash("Hurray! No Tumor Detected")
            return render_template('index.html')
    else:
        flash('Allowed image types are - tif, png, jpg, jpeg, gif')
        return redirect(request.url)
        
 
@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
if __name__ == "__main__":
    app.debug = True
    app.run()