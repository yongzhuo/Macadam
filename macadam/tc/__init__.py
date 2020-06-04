# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/4/29 23:43
# @author  : Mo
# @function: text-classification(tc)


from macadam.tc.t00_predict import ModelPredict
from macadam.tc.t00_trainer import trainer
from macadam.tc.t00_map import graph_map
from bert4keras.backend import set_gelu
set_gelu("tanh")  # "erf" or "tanh"

