# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/5/8 21:33
# @author  : Mo
# @function: test trainer


import os
# cpu-gpu与tf.keras
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_KERAS"] = "1"
from macadam.conf.path_config import path_root, path_tc_baidu_qa_2019, path_tc_thucnews
from macadam.tc import trainer


if __name__=="__main__":

    # bert-embedding地址, 必传
    path_embed = "D:/soft_install/dataset/word_embedding/sgns.wiki.word"
    # path_embed = "D:/soft_install/dataset/bert-model/chinese_L-12_H-768_A-12"
    path_checkpoint = path_embed + "/bert_model.ckpt"
    path_config = path_embed + "/bert_config.json"
    path_vocab = path_embed + "/vocab.txt"

    # 训练/验证数据地址
    # path_train = os.path.join(path_tc_baidu_qa_2019, "train.json")
    # path_dev = os.path.join(path_tc_baidu_qa_2019, "dev.json")
    path_base = "D:/workspace/pythonMyCode/Macadam/macadam/data/corpus/text_classification/baidu_qa_2019"
    path_train = os.path.join(path_base, "train.json")
    path_dev = os.path.join(path_base, "dev.json")
    # 网络结构, 嵌入模型, 大小写都可以
    # 网络模型架构(Graph), "Finetune", "FastText", "TextCNN", "CharCNN",
    # "BiRNN", "RCNN", "DCNN", "CRNN", "DeepMoji", "SelfAttention", "HAN", "Capsule"
    network_type = "RCNN"
    # 嵌入(embedding)类型, "ROOBERTA", "ELECTRA", "RANDOM", "ALBERT", "XLNET", "NEZHA", "GPT2", "WORD", "BERT"
    embed_type = "RANDOM"
    # token级别, 一般为"char", 只有random和word的embedding时存在"word"
    token_type = "WORD"
    # 任务, "TC", "SL", "RE"
    task = "TC"
    # 模型保存目录
    path_model_dir = os.path.join(path_root, "data", "model", network_type)
    # 开始训练, 可能前几轮loss较大acc较低, 后边会好起来
    trainer(path_model_dir, path_embed, path_train, path_dev, path_checkpoint, path_config, path_vocab,
            network_type=network_type, embed_type=embed_type, token_type=token_type, task=task,
            is_length_max=True, use_onehot=False, use_file=False, layer_idx=[-1],
            learning_rate=1e-3, batch_size=32,
            epochs=6, early_stop=3, rate=1)
    mm = 0

