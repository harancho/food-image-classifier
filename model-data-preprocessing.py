import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import pickle

DIRECTIORY = "/home/harsh/AndroidStudioProjects/Foodimagerecognization/food-image-classifier/dataset"

CATEGORIES = ['donuts','french_fries','fried_rice','ice_cream','pizza']

IMG_SIZE = 300

data =[]

for category in CATEGORIES:
	folder = os.path.join(DIRECTIORY,category)
	label = CATEGORIES.index(category)
	for image in os.listdir(folder):
		img_path = os.path.join(folder,image)
		img_array = cv2.imread(img_path)                #reading the RGB of image through cv2
		img_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))     #converting all images to same size so that we can process them
		data.append([img_array,label])
		
print(len(data))
random.shuffle(data)

x = []
y = []

for features,labels in data:
	x.append(features)        #adding all RGB to x list
	y.append(labels)          #adding all lables to y list

x = np.array(x)         #converting list to array through numpy
y = np.array(y)

pickle.dump(x,open('x.pkl','wb'))    #saving arrays in a file in pc
pickle.dump(y,open('y.pkl','wb'))