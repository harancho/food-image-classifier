import pickle
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
import time
from tensorflow.keras.callbacks import TensorBoard

NAME = 'cat-vs-dog-predictor-' + str(time.time())

tenserboard = TensorBoard(log_dir='./logs/'+NAME)

x = pickle.load(open('x.pkl','rb'))  #again getting back the saved arrays
y = pickle.load(open('y.pkl','rb'))

x = x/255   #sice RGB cannot exceed 255, we are dividing it by 255(feature sacling) for easy calculation

model = Sequential()   #instance created of NN class

model.add(Conv2D(64,(3,3),activation = 'relu'))   #using relu instead of sigmoid and 64 convolution each of size 3X3
model.add(MaxPooling2D((2,2)))                    #adding maxpolling layer of size 2X2   

model.add(Conv2D(64,(3,3),activation = 'relu'))   #adding 2nd convolution and maxpolling layer
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128,(3,3),activation = 'relu'))   #adding 2nd convolution and maxpolling layer
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128,(3,3),activation = 'relu'))   #adding 2nd convolution and maxpolling layer
model.add(MaxPooling2D((2,2)))

model.add(Flatten())              # i need to understand all this

print(x.shape)
model.add(Dense(512,input_shape = x.shape[1:],activation = 'relu'))

model.add(Dense(256,activation = 'relu'))

model.add(Dense(5,activation = 'softmax'))

model.compile(optimizer = 'adam' , loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(x,y,epochs = 30, validation_split = 0.1,batch_size = 32 )

model.save_weights('model_weights.h5')
model.save('model_keras.h5')