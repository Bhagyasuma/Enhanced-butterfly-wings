from flask import Flask, request, render_template
import os
import logging
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

try:
    model = load_model('vgg16_model.h5')
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error("Error loading model", exc_info=True)

labels = ['ADONIS', 'BLUE MORPHO', 'MONARCH', 'PURPLE HAIRSTREAK', 'SWALLOWTAIL']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400
    file = request.files['image']
    if file.filename == '':
        return "Empty file name", 400
    file_path = os.path.join('static', file.filename)
    file.save(file_path)
    img = load_img(file_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_label = labels[predicted_index]
    return render_template('result.html', label=predicted_label, filename=file.filename)

if __name__ == '__main__':
    app.run(debug=True)
