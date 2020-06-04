# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/4/26 20:24
# @author  : Mo
# @function: get output layer of bert4keras


from bert4keras.models import build_transformer_model
import json


path_embed_bert = "D:/soft_install/dataset/bert-model/chinese_L-12_H-768_A-12/"
config_path = path_embed_bert + "bert_config.json"
checkpoint_path = path_embed_bert + "bert_model.ckpt"
model = build_transformer_model(config_path,
                                checkpoint_path,)

configs = {}
if config_path:
    configs.update(json.load(open(config_path)))

config_model = model.get_config()
print(config_model)
num_hidden_layers = configs.get("num_hidden_layers")
outputs = []
for i in range(num_hidden_layers):
    output_layer = 'Transformer-{0}-FeedForward-Norm'.format(i)
    output = model.get_layer(output_layer).output
    outputs.append(output)
    mm = 0

mm = 0

