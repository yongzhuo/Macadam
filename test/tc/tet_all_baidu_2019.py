# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/6/30 21:47
# @author  : Mo
# @function:


# 适配linux
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(path_root)
# path
from macadam.conf.path_config import path_root, path_tc_baidu_qa_2019, path_tc_thucnews
# train
from macadam.tc import trainer
import os
# predict
from macadam.base.utils import txt_read, txt_write, load_json, save_json
from macadam.tc import ModelPredict
import json
## cpu-gpu与tf.keras
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_KERAS"] = "1"


if __name__=="__main__":
    # bert-embedding地址, 必传
    # path_embed = "D:/soft_install/dataset/word_embedding/sgns.wiki.word"
    # path_embed = "D:/soft_install/dataset/bert-model/chinese_L-12_H-768_A-12"
    print(os.environ["CUDA_VISIBLE_DEVICES"])

    # 训练/验证数据地址
    path_train = os.path.join(path_tc_baidu_qa_2019, "train.json")
    path_dev = os.path.join(path_tc_baidu_qa_2019, "dev.json")
    # path_train = os.path.join(path_tc_thucnews, "train.json")
    # path_dev = os.path.join(path_tc_thucnews, "dev.json")
    # 网络结构, 嵌入模型, 大小写都可以
    # 网络模型架构(Graph),
    network_types = ["FastText", "TextCNN", "CharCNN", "BiRNN", "RCNN", "CRNN",
                     "Finetune", "DCNN", "DeepMoji", "SelfAttention", "HAN", "Capsule"]
    embed_types = ["RANDOM", "WORD", "BERT"]
    token_types = ["CHAR", "WORD"] # ,"NGRAM"]


    nets = []
    for network_type in network_types:
        for embed_type in embed_types:
            for token_type in token_types:
                # if network_type=="Finetune":
                #     embed_type = "BERT"
                #     token_type = "CHAR"
                if embed_type == "BERT":
                    token_type = "CHAR"
                # if token_type=="NGRAM":
                #     embed_type = "RANDOM"
                net = [network_type, embed_type, token_type]
                if net not in nets:
                    nets.append(net)

    for [network_type, embed_type, token_type] in nets[18:19]: # nets[4:]:
        os.environ["MACADAM_LEVEL"] = "AUTO"
        print("pxy")
        print(os.environ["MACADAM_LEVEL"])

        print([network_type, embed_type, token_type])
        # path_embed
        path_embeds = ["D:/soft_install/dataset/word_embedding/sgns.wiki.word",
                       "D:/soft_install/dataset/bert-model/chinese_L-12_H-768_A-12"]

        path_embed = path_embeds[0] if embed_type in ["WORD", "RANDOM"] else path_embeds[1]
        path_checkpoint = path_embed + "/bert_model.ckpt"
        path_config = path_embed + "/bert_config.json"
        path_vocab = path_embed + "/vocab.txt"
        # 任务, "TC", "SL", "RE"
        task = "TC"
        # 模型保存目录
        path_model_dir = os.path.join(path_root, "data", "model", network_type + "_" + embed_type + "_" + token_type)
        # 学习率
        lr = 1e-5 if embed_type in ["ROBERTA", "ELECTRA", "ALBERT", "XLNET", "NEZHA", "GPT2", "BERT"] else 1e-3
        # embed-维度
        embed_size = 768 if embed_type in ["ROBERTA", "ELECTRA", "ALBERT", "XLNET", "NEZHA", "GPT2", "BERT"] else 300
        embed_size = 64 if token_type in "NGRAM" else embed_size
        early_stop = 1 if embed_type in ["ROBERTA", "ELECTRA", "ALBERT", "XLNET", "NEZHA", "GPT2", "BERT"] else 4
        print(os.environ["CUDA_VISIBLE_DEVICES"])
        # 开始训练
        time_collection = trainer(path_model_dir, path_embed, path_train, path_dev, path_checkpoint, path_config, path_vocab,
                                    network_type=network_type, embed_type=embed_type, token_type=token_type, task=task,
                                    is_length_max=False, use_onehot=False, use_file=True, layer_idx=[-1],
                                    early_stop=early_stop, learning_rate=lr, batch_size=32, epochs=32, rate=1)
        print("train ok!")
        # mp = ModelPredict(path_model_dir)
        # datas_dev = txt_read(path_dev)
        # datas_dev = [json.loads(dd.strip()) for dd in datas_dev]
        # metrics, report = mp.evaluate(datas_dev)
        # metrics["time"] = time_collection
        # print(json.dumps(metrics, ensure_ascii=False, indent=4))
        # print(report)
        # save_json(metrics, os.path.join(path_model_dir, "metrics.json"))
        # txt_write([report], os.path.join(path_model_dir, "metrics.txt"))
        mm = 0


# nohup python tet_all_baidu_2019_2.py > predict_batch_2.log 2>&1 &


# 注意, 一般而言数据量较少时, 训练集应该设置大于验证集
"""
Beforehand, slow generators could have caused race conditions and crashes with ValueError: generator already executing,
 e.g. if a validation generator filling up the queue took longer than a single epoch that elapsed meanwhile.

This sounds exotic but this happened to me in a generator that rendered PDF files and in the initial testing, 
epoch size was just a small multiply of batch size (and similar to validation set size). 
Of course this is not optimal usage, 
but nevertheless it was frustrating to debug this issue in initial model experiments.
"""


# bert-finetune-char/1-epcoh:     ,  ,
# bert-fasttext-char/1-epcoh:     2,  38210s, 0.8300
# bert-textcnn-char/1-epcoh:      3,  7980s,  0.8294
# bert-charcnn-char/1-epcoh:      2,  7800s,  0.8308
# bert-birnn-char/1-epcoh:
# bert-dcnn-char/1-epcoh:


# random-finetune-char/1-epcoh:   2,  275s,   0.6613
# random-fasttext-char/1-epcoh:   17, 280s,   0.7712
# random-charcnn-char/1-epcoh:    2,  370s,   0.7222
# random-textcnn-char/1-epcoh:    8,  570s,   0.7551
# random-birnn-char/1-epcoh:      2,  5200s,  0.7191
# random-dcnn-char/1-epcoh:       6,  3646s,  0.7536


# random-finetune-word/1-epcoh:   1,  2682s,  0.7348
# random-fasttext-word/1-epcoh:   2,  2520s,  0.7971
# random-charcnn-word/1-epcoh:    2,  2600s,  0.7946
# random-textcnn-word/1-epcoh:    2,  2800s,  0.7938
# random-birnn-word/1-epcoh:      1,  8393s,  0.7663
# random-dcnn-word/1-epcoh:       6,  3645s,  0.7536


# word-finetune-char/1-epcoh:     1,  230s,   0.7087
# word-fasttext-char/1-epcoh:     9,  260s,   0.7687
# word-charcnn-char/1-epcoh:      3,  330s,   0.7027
# word-textcnn-char/1-epcoh:      16, 368s,   0.7690
# word-birnn-char/1-epcoh:        1,  7360s,  0.7287
# word-dcnn-char/1-epcoh:         ,  ,  0.

# word-finetune-word/1-epcoh:     1,  1085s,  0.7797
# word-fasttext-word/1-epcoh:     2,  1082s,  0.8015
# word-charcnn-word/1-epcoh:      1,  1768s,  0.7968
# word-textcnn-word/1-epcoh:      3,  1210s,  0.7987
# word-birnn-word/1-epcoh:        2,  s,   0.
# wod-dcnn-word/1-epcoh:           ,  ,  0.


