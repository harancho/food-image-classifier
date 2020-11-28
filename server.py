from flask import Flask,jsonify,request,send_from_directory,abort,send_file
import os
from PIL import Image,ImageDraw
import time
import matplotlib.pyplot as plt
import cv2

app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

@app.route('/detect', methods = ['GET','POST'])
def detect():
	if request.method == 'POST':
		file = request.files['file']
		file = file.read()
		path = os.path.join("uploads", 'detect.jpeg')   # wrong approach
		with open(path,'wb') as f:                      # find sol for this
			f.write(file)
		
		frame = cv2.imread("uploads/detect.jpeg")
		plt.imshow(frame)
		plt.show()
		print('file uploaded for detection')
		return jsonify(result = 'ans')

	return jsonify(result = 'get request')

if __name__ == '__main__':
    app.run(debug=True)