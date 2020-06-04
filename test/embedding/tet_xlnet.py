# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/4/27 21:00
# @author  : Mo
# @function: test embedding of bert-like pre-train model


import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(path_root)
# cpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_KERAS"] = "1"
from macadam.base.embedding import *


def _tet_xlnet_embedding():
    ######## tet embed-xlnet
    path_embed = "D:/soft_install/dataset/bert-model/chinese_xlnet_mid_L-24_H-768_A-12"
    path_check_point = path_embed + "/xlnet_model.ckpt"
    path_config = path_embed + "/xlnet_config.json"
    path_vocab = path_embed + "/spiece.model"
    attention_type = "bi"
    memory_len = 16
    target_len = 32
    params = {"embed":{"path_embed":path_embed,
                       "layer_idx": [-1], # [-3, -2, -1],
                       "memory_len": memory_len,
                       "target_len": target_len,
                       "attention_type": attention_type},
              "sharing": {"length_max": target_len,
                          "trainable": False},
              }
    xlnet_embed = XlnetEmbedding(params)
    xlnet_embed.build_embedding()
    xlnet_embed.build_embedding(path_checkpoint=path_check_point,
                                path_config=path_config,
                                path_vocab=path_vocab,
                                attention_type=attention_type,
                                memory_len=memory_len,
                                target_len=target_len)
    return xlnet_embed


if __name__ == '__main__':
    embed = _tet_xlnet_embedding()
    res = embed.encode(text="macadam怎么翻译", second_text="macadam是碎石路")
    print(res)
    while True:
        print("请输入first_text:")
        first_text = input()
        print("请输入second_text:")
        second_text = input()
        res = embed.encode(text=first_text, second_text=second_text)
        print(res)
    mm = 0


