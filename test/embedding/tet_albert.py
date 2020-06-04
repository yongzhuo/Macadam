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


def _tet_albert_embedding():
    ###### tet albert
    # path_embed = "D:/soft_install/dataset/bert-model/google_albert/albert_base_v2"
    # path_check_point = path_embed + "/model.ckpt-best"
    # path_config = path_embed + "/albert_config.json"
    # path_vocab = path_embed + "/30k-clean.vocab"
    path_embed = "D:/soft_install/dataset/bert-model/albert_large_zh"
    path_check_point = path_embed + "/bert_model.ckpt"
    path_config = path_embed + "/bert_config.json"
    path_vocab = path_embed + "/vocab.txt"
    params = {"embed": {"path_embed": path_embed,
                        "layer_idx": [-1],
                        },
              "sharing": {"length_max": 512},
              }
    albert_embed = AlBertEmbedding(params)
    albert_embed.build_embedding(path_checkpoint=path_check_point,
                                 path_config=path_config,
                                 path_vocab=path_vocab)
    return albert_embed


if __name__ == '__main__':

    embed = _tet_albert_embedding()

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


