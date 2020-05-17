#importing the model-building libraries
import numpy as np                       
import tensorflow
import keras
from keras.models import Sequential

#initializing the model
cnn_model=Sequential()

#preprocessing data
from keras.layers import Dropout,Dense,Conv2D,MaxPooling2D,Flatten
from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)


#path of training data:D:\AAA\COURSES\AI\PROJECT\data
x_train=train_datagen.flow_from_directory(r"D:\Project_AI\data\data\trainset",target_size=(64,64),batch_size=32,class_mode='categorical')
x_test=train_datagen.flow_from_directory(r"D:\Project_AI\data\data\testset",target_size=(64,64),batch_size=32,class_mode='categorical')

#input layers
cnn_model.add(Conv2D(32,3,3,input_shape=(64,64,3),activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))
#cnn_model.add(Dropout(0.25))
cnn_model.add(Flatten())

#hidden layers
cnn_model.add(Dense(128,activation='relu'))

#output layer
cnn_model.add(Dense(5,activation='softmax'))

#configuring the learning process
cnn_model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])

#training the model
cnn_model.fit_generator(x_train,
                         samples_per_epoch = 5000,
                         nb_epoch = 5,
                         validation_data = x_test,
                         nb_val_samples = 1500)

#save the model
cnn_model.save('D:\Project_AI\model_op\crop_protection_model.h5')  #use this location for prediction  
                         
