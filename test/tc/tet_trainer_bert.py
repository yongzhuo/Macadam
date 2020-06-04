# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/5/8 21:33
# @author  : Mo
# @function: test trainer of bert


# 适配linux
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(path_root)
# cpu-gpu与tf.keras
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_KERAS"] = "1"
# macadam
from macadam.conf.path_config import path_root, path_tc_baidu_qa_2019, path_tc_thucnews
from macadam.tc import trainer
import os


if __name__=="__main__":
    # bert-embedding地址, 必传
    path_embed = "D:/soft_install/dataset/bert-model/chinese_L-12_H-768_A-12"
    path_checkpoint = path_embed + "/bert_model.ckpt"
    path_config = path_embed + "/bert_config.json"
    path_vocab = path_embed + "/vocab.txt"

    # 训练/验证数据地址
    # path_train = os.path.join(path_tc_thucnews, "train.json")
    # path_dev = os.path.join(path_tc_thucnews, "dev.json")
    path_train = os.path.join(path_tc_baidu_qa_2019, "train.json")
    path_dev = os.path.join(path_tc_baidu_qa_2019, "dev.json")

    # 网络结构, 嵌入模型, 大小写都可以
    # 网络模型架构(Graph), "FineTune", "FastText", "TextCNN", "CharCNN",
    # "BiRNN", "RCNN", "DCNN", "CRNN", "DeepMoji", "SelfAttention", "HAN", "Capsule"
    network_type = "TextCNN"
    # 嵌入(embedding)类型, "ROOBERTA", "ELECTRA", "RANDOM", "ALBERT", "XLNET", "NEZHA", "GPT2", "WORD", "BERT"
    embed_type = "BERT"
    # token级别, 一般为"char", 只有random和word的embedding时存在"word"
    token_type = "CHAR"
    # 任务, "TC"(文本分类), "SL"(序列标注), "RE"(关系抽取)
    task = "TC"
    # 模型保存目录
    path_model_dir = os.path.join(path_root, "data", "model", network_type)
    # 开始训练, 可能前几轮loss较大acc较低, 后边会好起来
    trainer(path_model_dir, path_embed, path_train, path_dev, path_checkpoint, path_config, path_vocab,
            network_type=network_type, embed_type=embed_type, token_type=token_type, task=task)
    mm = 0

