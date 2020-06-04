# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/4/27 21:00
# @author  : Mo
# @function: test embedding of bert-like pre-train model


import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(path_root)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_KERAS"] = "1"

from macadam.base.embedding import *
import os


def _tet_bert_embedding():
    # ###### tet bert
    path_embed = "D:/soft_install/dataset/bert-model/chinese_L-12_H-768_A-12"
    path_check_point = path_embed + "/bert_model.ckpt"
    path_config = path_embed + "/bert_config.json"
    path_vocab = path_embed + "/vocab.txt"
    params = {"embed": {"path_embed": path_embed,
                        "layer_idx": [-1],
                        },
              "sharing": {"length_max": 512},
              }
    bert_embed = BertEmbedding(params)
    bert_embed.build_embedding(path_checkpoint=path_check_point,
                               path_config=path_config,
                               path_vocab=path_vocab)
    return bert_embed


if __name__ == '__main__':
    embed = _tet_bert_embedding()
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


