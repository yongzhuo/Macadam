# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/4/22 21:13
# @author  : Mo
# @function: init macadam([məˈkædəm]) of tf_keras


import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(path_root)
from distutils.util import strtobool


# gpu/tf日志的环境, 默认CPU
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "-1") # "0,1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# 自动化(默认AUTO), 即定义是不是高自由度, 如"CUSTOM"可以高度自定义, 网络架构embedding/graph/loss等均可高度自定义
os.environ["MACADAM_LEVEL"] = os.environ.get("MACADAM_LEVEL", "AUTO")
# 默认使用tf.keras
tf_keras = os.environ.get("TF_KERAS", "1")
TF_KERAS = strtobool(tf_keras)
if TF_KERAS:
    import tensorflow.keras.backend as K
    import tensorflow.keras as keras
else:
    import keras.backend as K
    import keras


from bert4keras.optimizers import custom_objects as custom_objects_macadam_bert4keras_o
from bert4keras.backend import custom_objects as custom_objects_macadam_bert4keras_k
from bert4keras.layers import custom_objects as custom_objects_macadam_bert4keras_l
# from macadam.base.layers import custom_objects_macadam
# from keras_xlnet import get_custom_objects
# custom_objects_xlnet = get_custom_objects()

keras.utils.get_custom_objects().update(custom_objects_macadam_bert4keras_o)
keras.utils.get_custom_objects().update(custom_objects_macadam_bert4keras_l)
keras.utils.get_custom_objects().update(custom_objects_macadam_bert4keras_k)
# keras.utils.get_custom_objects().update(custom_objects_macadam)
# keras.utils.get_custom_objects().update(custom_objects_xlnet)


O = keras.optimizers
C = keras.callbacks
L = keras.layers
M = keras.models


__version__ = "0.0.1"

