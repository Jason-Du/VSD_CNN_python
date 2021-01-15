from solve_cudnn_error import *
solve_cudnn_error()
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model
import numpy as np
from keras.models import Model
from keras.datasets import cifar10
import sys
import os
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
x_train = X_train.astype('float32') / 255
print(Y_train[0])
os.system('pause')
# print(x_train)
# print(x_train)
np.set_printoptions(threshold=np.inf)
y_train = np_utils.to_categorical(Y_train)
model = load_model('VSD.h5')

pre = model.evaluate(x_train[0:2000,:,:,:], y_train[0:2000,:], batch_size=64, verbose=2)
print('test_loss:', pre[0], '- test_acc:', pre[1])
# os.system('pause')
# model =load_model("keras_model.h5")
for idx in range(len(model.layers)):
  print(model.get_layer(index = idx).name)
  # max_pooling2d_2
  # flatten_1
# test=np.random.rand(32,32,3)[np.newaxis,...]
test_index=0
test=x_train[test_index,:,:,:][np.newaxis,...]
# print(test.shape)
# test=np.ones((32,32,3))[np.newaxis,...]
# print(test)
layer_name = 'flatten_1'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(test)
print(intermediate_output)
print(intermediate_output.shape)
print("**********************************************************************************************************************")
# with np.printoptions(threshold=np.inf):
#     print(np.squeeze(intermediate_output)[:,:,0])

# np.save("maxpooling_output",np.squeeze(intermediate_output))
# print(np.squeeze(intermediate_output).shape)
print("**********************************************************************************************************************")
layer_name = 'conv2d_4'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output1 = intermediate_layer_model.predict(test)
print(intermediate_output1)
print(intermediate_output1.shape)
# np.save("flatten_output",np.squeeze(intermediate_output1))

# with np.printoptions(threshold=np.inf):
#     print(np.squeeze(intermediate_output1))
print(Y_test[test_index,:])




# inp = model.input                                           # input placeholder
# outputs = [layer.output for layer in model.layers]          # all layer outputs
# functors = [K.function([inp], [out]) for out in outputs]    # evaluation functions
#
# # Testing
# test = np.random.rand(32,32,3)[np.newaxis,...]
# layer_outs = [func([test]) for func in functors]
# print (layer_outs)
# conv2d_1
# conv2d_2
# max_pooling2d_1
# conv2d_3
# conv2d_4
# max_pooling2d_2
# flatten_1
# dense_1