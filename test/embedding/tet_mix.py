# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/5/15 21:20
# @author  : Mo
# @function: test embedding of pre-train model of mix word and char


import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(path_root)
# cpu-gpu与tf.keras
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_KERAS"] = "1"
os.environ["MACADAM_LEVEL"] = "custom" # 自定义, 只要不是auto就好
from macadam.conf.path_config import path_tc_baidu_qa_2019
from macadam.base.embedding import *
import os


def _tet_mix_embedding():
    ######## tet embed-roberta

    # 训练/验证数据地址
    # path_train = os.path.join(path_tc_thucnews, "train.json")
    # path_dev = os.path.join(path_tc_thucnews, "dev.json")
    path_train = os.path.join(path_tc_baidu_qa_2019, "train.json")
    path_dev = os.path.join(path_tc_baidu_qa_2019, "dev.json")
    # params
    params_word = {
              "embed": {"path_embed": None,
                        },
              "sharing": {"length_max": 32,     # 句子最大长度, 不配置则会选择前95%数据的最大长度, 配置了则会强制选择, 固定推荐20-50, bert越长会越慢, 占用空间也会变大, 小心OOM
                          "embed_size": 64,    # 字/词向量维度, bert取768, word取300, char可以更小些
                          "task": "TC",          # 任务类型, TC是原始, SL则加上CLS,SEP。"SL"(sequence-labeling), "TC"(text-classification),"RE"(relation-extraction)
                          "token_type": "word",  # 级别, 最小单元, 字/词, 填 "char" or "word", "ngram", 注意:word2vec模式下训练语料要首先切好
                          "embed_type": "random", # 级别, 嵌入类型, 还可以填"word2vec"、"random"、 "bert"、 "albert"、"roberta"、"nezha"、"xlnet"、"electra"、"gpt2"
                          },
              "data": {"train_data": path_train,  # 训练数据
                       "val_data": path_dev  # 验证数据
                       },
              }
    params_char = {
              "embed": {"path_embed": None,
                        },
              "sharing": {"length_max": 32,      # 句子最大长度, 不配置则会选择前95%数据的最大长度, 配置了则会强制选择, 固定推荐20-50, bert越长会越慢, 占用空间也会变大, 小心OOM
                          "embed_size": 64,      # 字/词向量维度, bert取768, word取300, char可以更小些
                          "task": "TC",           # 任务类型, TC是原始, SL则加上CLS,SEP。"SL"(sequence-labeling), "TC"(text-classification),"RE"(relation-extraction)
                          "token_type": "char",   # 级别, 最小单元, 字/词, 填 "char" or "word", "ngram", 注意:word2vec模式下训练语料要首先切好
                          "embed_type": "random", # 级别, 嵌入类型, 还可以填"word2vec"、"random"、 "bert"、 "albert"、"roberta"、"nezha"、"xlnet"、"electra"、"gpt2"
                          },
              "data": {"train_data": path_train,  # 训练数据
                       "val_data": path_dev       # 验证数据
                       },
              }
    params = [params_char, params_word]
    word2vec_embed = MixEmbedding(params)
    word2vec_embed.build_embedding()
    return word2vec_embed


if __name__ == '__main__':

    embed = _tet_mix_embedding()

    res = embed.encode(text="北京在哪里来着", second_text=["macadam是碎石路"], use_seconds=True)
    print(res)
    while True:
        print("请输入first_text:")
        first_text = input()
        print("请输入second_text:")
        second_text = input()
        res = embed.encode(text=first_text, second_text=second_text)
        print(res)

    mm = 0


text = [['[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]'], ['北', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]'], ['北京', '京', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]'], ['在', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]'], ['哪', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]'], ['哪里', '里', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]'], ['来', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]']]

second_texts = [[['[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]'], ['m', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]'], ['ma', 'a', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]'], ['mac', 'ac', 'c', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]'], ['aca', 'ca', 'a', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]'], ['acad', 'cad', 'ad', 'd', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]'], ['ada', 'da', 'a', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]'], ['am', 'm', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]'], ['是', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]'], ['碎', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]'], ['碎石', '石', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]', '[PAD-WC]']]]


