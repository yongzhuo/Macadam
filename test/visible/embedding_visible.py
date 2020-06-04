# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/5/2 17:18
# @author  : Mo
# @function: embedding的visible， 即可视化


# 加入搜索根目录, 适配linux/win10等的非编程开发工具情况
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(path_root)
os.environ["TF_KERAS"] = "1"

from macadam.base.embedding import BertEmbedding
from pylab import *

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import json


###### tet bert
path_embed = "D:/soft_install/dataset/bert-model/chinese_L-12_H-768_A-12"
path_check_point = path_embed + "/bert_model.ckpt"
path_config = path_embed + "/bert_config.json"
path_vocab = path_embed + "/vocab.txt"
params = {"embed": {"path_embed": path_embed,
                    "layer_idx": [-2],
                    },
          "sharing": {"length_max": 128},
          }
bert_embed = BertEmbedding(params)
bert_embed.build_embedding(path_checkpoint=path_check_point,
                           path_config=path_config,
                           path_vocab=path_vocab)

def cosine(sen_1, sen_2):
    """
        余弦距离
    :param sen_1: numpy.array
    :param sen_2: numpy.array
    :return: float, like 0.0
    """
    if sen_1.all() and sen_2.all():
        return np.dot(sen_1, sen_2) / (np.linalg.norm(sen_1) * np.linalg.norm(sen_2))
    else:
        return 0.0

def bert_plt(reses, first_text, second_text):
    def show(similarity_cls, first_text, second_text):
        mpl.rcParams["font.sans-serif"] = ["SimHei"]
        x = [i for i in range(len(similarity_cls))]
        labels = ["[CLS]"] + [i for i in first_text] + ["[SEP]"] + [i for i in second_text] + ["[SEP]"]
        y = similarity_cls
        plt.xticks(x, labels) # 替换标签
        plt.plot(x, y, 'o-', label=u"线条")  # 画图
        plt.show()

    for res in reses:
        similarity_cls = []
        res_i = np.around(res.tolist(), decimals=16).tolist()
        res_return = res_i
        tensor_cls = np.array(res_return[0])
        tensor_sen = np.array(res_return[:len(first_text + second_text) + 3])
        for ts in tensor_sen:
            cos_cls = cosine(tensor_cls, ts)
            similarity_cls.append(cos_cls)
        show(similarity_cls, first_text, second_text)


def bert_plt_two(cls, res):
    def show(similarity_cls, first_text):
        mpl.rcParams["font.sans-serif"] = ["SimHei"]
        x = [i for i in range(len(similarity_cls))]
        labels = ["[CLS]"] + [i for i in first_text] + ["[SEP]"]
        y = similarity_cls
        plt.xticks(x, labels) # 替换标签
        plt.plot(x, y, 'o-', label=u"线条")  # 画图
        plt.show()

    similarity_cls = []
    cls = np.around(cls.tolist(), decimals=16)
    res_i = np.around(res.tolist(), decimals=16).tolist()
    res_return = res_i
    tensor_sen = np.array(res_return)
    for ts in tensor_sen:
        cos_cls = cosine(cls, ts[0])
        similarity_cls.append(cos_cls)
    show(similarity_cls, first_text)


first_text = "北京天安门广场上空"
second_text = None
res = bert_embed.encode(text=first_text, second_text=second_text)
bert_plt(res, first_text, second_text)


cls_0 = res[0][0]
first_text = "祖冲之是中国伟大的数学家，他创建的割元术直到21世纪的今天仍然有现实意义"
second_text = None
res2 = bert_embed.encode(text=first_text, second_text=second_text)
print(cosine(res[0][0], res2[0][0]))


bert_plt_two(cls_0, res2[0])

cls_0 = res[0][0] # np.around(res[0].tolist(), decimals=16)
cls_2 = res2[0][0] # np.around(res2[0].tolist(), decimals=16)
print(cosine(cls_0, cls_2))

while True:
    print("请输入first_text:")
    first_text = input()
    print("请输入second_text:")
    second_text = input()
    res = bert_embed.encode(text=first_text, second_text=second_text)
    bert_plt(res, first_text, second_text)

