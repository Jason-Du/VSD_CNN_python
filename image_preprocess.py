import cv2
from keras.datasets import mnist
import numpy as np

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print(X_train.shape)
print(Y_train.shape)
# X_train=X_train[...,np.newaxis]
test1=X_train[0]
print(test1.shape)
row_zero=np.zeros((2,28))
col_zero=np.zeros((32,2))
# A=np.vstack([row_zero,row_zero])
# print(A.shape)
padding_list=[]
for single_train_data in X_train:
    row_padding=np.vstack([row_zero,single_train_data,row_zero])
    col_padding=np.hstack([col_zero,row_padding,col_zero])
    channel_padding=np.stack([col_padding,col_padding,col_padding],axis=2)
    padding_list.append(channel_padding)
A=np.stack(padding_list,axis=0)
print(A.shape)

padding_list=[]
for single_train_data in X_test:
    row_padding=np.vstack([row_zero,single_train_data,row_zero])
    col_padding=np.hstack([col_zero,row_padding,col_zero])
    channel_padding=np.stack([col_padding,col_padding,col_padding],axis=2)
    padding_list.append(channel_padding)
B=np.stack(padding_list,axis=0)
print(B.shape)
np.save("MNIST_padding_trainx.npy",A)
np.save("MNIST_padding_testx.npy",B)