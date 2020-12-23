import os
# from solve_cudnn_error import *
# solve_cudnn_error()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Input,Dense,Conv2D,Flatten,MaxPooling2D,Activation,Dropout,AveragePooling2D,ZeroPadding2D,concatenate,add,GlobalAveragePooling2D,Conv1D,MaxPooling1D
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler

def CNN(width, height, depth, classes):
    # initialize the model
    model = Sequential()

    model.add(Conv2D(input_shape=(width, height, depth), kernel_size=(3, 3), filters=8, strides=(1,1), activation='relu',padding="valid"))
    model.add(Conv2D(data_format="channels_last", kernel_size=(3, 3), filters=8, strides=(1, 1), activation='relu',padding="valid"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
    model.add(Conv2D(data_format="channels_last", kernel_size=(3, 3), filters=8, strides=(1, 1), activation='relu',padding="valid"))
    model.add(Conv2D(data_format="channels_last", kernel_size=(3, 3), filters=8, strides=(1, 1), activation='relu',padding="valid"))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
    # model.add(Conv2D(data_format="channels_last", kernel_size=(3, 3), filters=8, strides=(1, 1), activation='relu',padding="valid"))
    # model.add(Conv2D(data_format="channels_last", kernel_size=(3, 3), filters=8, strides=(1, 1), activation='relu',padding="valid"))
    model.add(Flatten(data_format="channels_last"))
    # model.add(Dense(128, activation='relu'))
    model.add(Dense(classes, activation='softmax'))
    return model
if __name__ == '__main__':
    from keras.datasets import cifar10
    from keras.utils import np_utils, plot_model
    from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    x_train = X_train.astype('float32') / 255
    x_test = X_test.astype('float32') / 255
    np.save("testx",x_train)
    np.save("testy", Y_train)
    y_train = np_utils.to_categorical(Y_train)
    y_test = np_utils.to_categorical(Y_test)
    print(y_train.shape)
    print(x_train.shape)
    print(y_test.shape)
    print(x_test.shape)
    model=CNN(32,32,3,10)
    model.summary()
    model.compile(optimizer=Adam(lr=0.0004,beta_1=0.9, beta_2=0.999, epsilon=1e-08),loss='categorical_crossentropy', metrics=['accuracy'])
    History = model.fit(x_train, y_train, batch_size=128, epochs=500, verbose=2, validation_data=(x_test, y_test))
    pre = model.evaluate(x_test, y_test, batch_size=64, verbose=2)

    # print('test_loss:', pre[0], '- test_acc:', pre[1])
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(History.history['accuracy'])
    plt.plot(History.history['val_accuracy'])
    plt.title('DNN accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.subplot(1, 2, 2)
    plt.plot(History.history['loss'])
    plt.plot(History.history['val_loss'])
    plt.title('DNN loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.show()
    os.system('pause')
    model.save('VSD.h5')