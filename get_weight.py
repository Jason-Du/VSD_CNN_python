from keras.models import load_model

import numpy as np
import os
model = load_model('VSD.h5')
for index_layer,layer in enumerate(model.layers):
    weights = layer.get_weights()
    for sinlge_weight_bias_index,sinlge_weight_bias in enumerate(weights):
        np.save("./weight_folder/layer{}_weight{}".format(index_layer,sinlge_weight_bias_index),sinlge_weight_bias)
        # print(index_layer)
        # print(sinlge_weight_bias.shape)
        # print(type(sinlge_weight_bias))
        # print(sinlge_weight_bias)
#

