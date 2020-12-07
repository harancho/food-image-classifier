from flask import Flask,jsonify,request,send_from_directory,abort,send_file
import os
from PIL import Image,ImageDraw
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

DIRECTORY = "/home/harsh/AndroidStudioProjects/Foodimagerecognization/food-image-classifier"
model = tf.keras.models.load_model(DIRECTORY + '/model_keras_18.h5')

@app.route('/detect', methods = ['GET','POST'])
def detect():
	if request.method == 'POST':
		file = request.files['file']
		file = file.read()
		path = os.path.join("uploads", 'detect.jpeg')   # wrong approach
		with open(path,'wb') as f:                      # find sol for this
			f.write(file)
		
		img_array =[]
		data = cv2.imread("uploads/detect.jpeg")
		img_array.append(data)
		img_array = np.array(img_array)
		img_array = img_array/255
		# plt.imshow(frame)
		# plt.show()
		output = model.predict_classes(img_array,verbose = 2)
		print(output)
		if output[0] == 0:
			ans = 'Chocolate Cake'
		if output[0] == 1:
			ans = "Donut"
		if output[0] == 2:
			ans = "French Fries"
		if output[0] == 3:
			ans = "Fried Rice"
		if output[0] == 4:
			ans = "Hotdog"

		print(ans)
		print('file uploaded for detection')
		return jsonify(result = ans)

	return jsonify(result = 'get request')

if __name__ == '__main__':
    app.run(debug=True)