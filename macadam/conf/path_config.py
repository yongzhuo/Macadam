# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/4/22 21:13
# @author  : Mo
# @function: base path of Macadam


# 适配linux
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(path_root)


# path of embedding, 最好是使用os.path, 不要使用"/","\\"等, 词向量-字向量-字(全-wiki)-BERT
path_embed_word2vec_word = os.path.join(path_root, "data", "embed", "sgns.wiki.word")
path_embed_word2vec_char = os.path.join(path_root, "data", "embed", "sgns.wiki.char")
path_embed_random_char = os.path.join(path_root, "data", "embed", "default_character.txt")
path_embed_bert = os.path.join(path_root, "data", "embed", "chinese_L-12_H-768_A-12")
# corpus of text-classification
path_tc_baidu_qa_2019 = os.path.join(path_root, "data", "corpus", "text_classification", "baidu_qa_2019")
path_tc_thucnews = os.path.join(path_root, "data", "corpus", "text_classification", "thucnews")
# corpus of name-entity-recognition
path_ner_people_1998 = os.path.join(path_root, "data", "corpus", "sequence_labeling", "ner_people_1998")
path_ner_clue_2020 = os.path.join(path_root, "data", "corpus", "sequence_labeling", "ner_clue_2020")

# 模型目录
path_model_dir = os.path.join(path_root, "data", "model", "base")
# 超参数保存地址
path_model_info = os.path.join(path_root, "data", "model", "base", "macadam.info")
# embedding微调保存地址
path_fineture = os.path.join(path_root, "data", "model", "base", "macadam.embed")
# 模型保存地址
path_model = os.path.join(path_root, "data", "model", "base", "macadam.h5")
