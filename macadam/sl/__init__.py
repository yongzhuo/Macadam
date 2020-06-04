# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/4/29 23:43
# @author  : Mo
# @function: sequence-labeling(ner, pos, cws, tag)


from macadam.sl.s00_predict import ModelPredict
from macadam.sl.s00_trainer import trainer
from macadam.sl.s00_map import graph_map
from bert4keras.backend import set_gelu
set_gelu("tanh")  # "erf" or "tanh"

