# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/5/15 21:20
# @author  : Mo
# @function: test embedding of pre-train model of word2vec


import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(path_root)
# cpu-gpu与tf.keras
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["MACADAM_LEVEL"] = "custom" # 自定义, 只要不是auto就好
os.environ["TF_KERAS"] = "1"
from macadam.conf.path_config import path_tc_baidu_qa_2019, path_tc_thucnews
from macadam.base.embedding import *
import os


def _tet_word2vec_embedding():
    ######## tet embed-roberta

    # 训练/验证数据地址
    path_train = os.path.join(path_tc_thucnews, "train.json")
    path_dev = os.path.join(path_tc_thucnews, "dev.json")
    # path_train = os.path.join(path_tc_baidu_qa_2019, "train.json")
    # path_dev = os.path.join(path_tc_baidu_qa_2019, "dev.json")
    # params
    params = {
              # "embed": {"path_embed": None,
              #           },
              "sharing": {"length_max": 128,     # 句子最大长度, 不配置则会选择前95%数据的最大长度, 配置了则会强制选择, 固定推荐20-50, bert越长会越慢, 占用空间也会变大, 小心OOM
                          "embed_size": 300,    # 字/词向量维度, bert取768, word取300, char可以更小些
                          "task": "TC",          # 任务类型, TC是原始, SL则加上CLS,SEP。"SL"(sequence-labeling), "TC"(text-classification),"RE"(relation-extraction)
                          "token_type": "word",  # 级别, 最小单元, 字/词, 填 "char" or "word", "ngram", 注意:word2vec模式下训练语料要首先切好
                          "embed_type": "random", # 级别, 嵌入类型, 还可以填"word2vec"、"random"、 "bert"、 "albert"、"roberta"、"nezha"、"xlnet"、"electra"、"gpt2"
                          },
              "data": {"train_data": path_train,  # 训练数据
                       "val_data": path_dev  # 验证数据
                       },
              }
    word2vec_embed = RandomEmbedding(params)
    word2vec_embed.build_embedding(path_corpus=path_train)
    return word2vec_embed


if __name__ == '__main__':

    embed = _tet_word2vec_embedding()

    res = embed.encode(text="北京在哪里来着", second_text=["macadam是碎石路"], use_seconds=True)
    # print(res)
    while True:
        print("请输入first_text:")
        first_text = input()
        print("请输入second_text:")
        second_text = input()
        res = embed.encode(text=first_text, second_text=second_text)
        print(res)

    mm = 0

