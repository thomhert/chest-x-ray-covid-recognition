# By Thomhert S. Siadari
# @thomhert
# thomhert.ss@gmail.compile
# ======================================

#::: Import modules and packages :::
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Import Keras dependencies
# from tensorflow.keras.models import model_from_json
# from tensorflow.python.framework import ops
# ops.reset_default_graph()
import keras
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.xception import preprocess_input


# Import other dependecies
import numpy as np
import h5py
from PIL import Image
import PIL
import os
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

#::: Flask App Engine :::
# Define a Flask app
app = Flask(__name__)

# ::: Prepare Keras Model :::
# Model files
#MODEL_ARCHITECTURE = './model/model_adam.json'
MODEL_WEIGHTS = './model/best_xception_20200316.hdf5'

model = load_model(MODEL_WEIGHTS, compile=False)
t_graph = tf.get_default_graph()

# Load the model from external files
#json_file = open(MODEL_ARCHITECTURE)
#loaded_model_json = json_file.read()
#json_file.close()
# model = model_from_json(loaded_model_json)


# Get weights into the model
print('Model loaded. Check http://127.0.0.1:5000/')


# ::: MODEL FUNCTIONS :::
def model_predict(img_path, model):
	'''
		Args:
			-- img_path : an URL path where a given image is stored.
			-- model : a given Keras CNN model.
	'''

	IMG = image.load_img(img_path, target_size=(150, 150))
	# Pre-processing the image
	IMG_ = image.img_to_array(IMG)
	IMG = np.expand_dims(IMG_, axis=0)
	print(IMG.shape)
	IMG = preprocess_input(IMG)

	# print(model)
	# model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
	with t_graph.as_default():
		prediction = model.predict(IMG)

	return prediction


# ::: FLASK ROUTES
@app.route('/', methods=['GET'])
def index():
	# Main Page
	return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():

	# Constants:

	if request.method == 'POST':

		# Get the file from post request
		f = request.files['file']

		# Save the file to ./uploads
		basepath = os.path.dirname(__file__)
		file_path = os.path.join(
			basepath, 'uploads', secure_filename(f.filename))
		f.save(file_path)

		# Make a prediction
		prediction = model_predict(file_path, model)
		pred_class_index = np.argmax(prediction)
		print(prediction[0][pred_class_index])
		predicted_class = "COVID" if pred_class_index == 0 else "NORMAL"
		print('We think that is {} with accuracy {}%.'.format(predicted_class, prediction[0][pred_class_index]*100))

		return str(predicted_class) + ' (' + str(prediction[0][pred_class_index]*100) + '%)'

if __name__ == '__main__':
	app.run(debug = True)