# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/4/26 20:25
# @author  : Mo
# @function: constant of token-symbol and hyper-parameters-default


from macadam.conf.path_config import path_model_dir
from typing import Dict
import os


EMBEDDING_TYPE = ["ROBERTA","ELECTRA","RANDOM","ALBERT",
                   "XLNET","NEZHA","GPT2","WORD","BERT", "MIX"]


# symbol of common token
MASK = "[MASK]"
CLS = "[CLS]"
SEP = "[SEP]"
PAD = "[PAD]"
UNK = "[UNK]"
BOS = "[BOS]"
EOS = "[EOS]"
WC = "[WC]"


# task of macadam
SL = "SL" # sequence-labeling(ner, pos, tag)
TC = "TC" # text-classification
RE = "RE" # relation-extraction


# hyper_parameters of deep-learning, include sharing, embed, graph, train, save and data
hyper_parameters_default = {
"sharing": {"length_max": None,              # 句子最大长度, 不配置则会选择前95%数据的最大长度, 配置了则会强制选择, 固定推荐20-50, bert越长会越慢, 占用空间也会变大, 小心OOM
            "embed_size": 768,               # 字/词向量维度, bert取768, word取300, char可以更小些
            "vocab_size": None,              # 字典/词典大小, 可根据具体语料更新, 可不配置
            "trainable": True,               # embedding是静态的还是动态的, 即控制可不可以微调
            "task": None,                    # 任务类型, "SL"(sequence-labeling), "TC"(text-classification),"RE"(relation-extraction)
            "token_type": "CHAR",            # 级别, 最小单元, 字/词, 填 "CHAR" or "WORD", "NGRAM", 注意:word2vec模式下训练语料要首先切好
            "embed_type": "BERT",            # 级别, 嵌入类型, 还可以填"WORD"、"RANDOM"、 "BERT"、 "ALBERT"、"ROBERTA"、"NEZHA"、"XLNET"、"ELECTRA"、"GPT2"
            "gpu_memory_fraction": 0.60,     # gpu使用率, 0-1
             },
"embed": {"layer_idx": [-2],                 # 取bert的layer层输出, -1~-12, 0-11等, eg. 0, 1, 11, -1, -2, -12等
          "path_embed": None,                # 外部embed模型地址, 如word2vec, bert
          "merge_type": "concat",            # bert的layer层输出融合方式, 包括 "concat", "add", "pool-max", "pool-avg", "multi"
          "application": "encode",            # bert4keras下游任务, "encode", "lm", "unilm"等
          "length_first": None,              # 第一句最大长度, 大则截断-小则padding
          "length_second": None,             # 第二句最大长度, 大则截断-小则padding
          "xlnet_embed": {"attention_type": "bi",
                          "memory_len": 0,
                          "target_len": 5},  # xlnet的参数, 使用的是keras-xlnet
          },
"graph": {"filters_size": [3, 4, 5],         # 卷积核尺寸, 1-10
          "filters_num": 300,                # 卷积个数 text-cnn:300-600
          "rnn_type": None,                # 循环神经网络, select "LSTM", "GRU", "Bidirectional-GRU"
          "rnn_unit": 256,                   # RNN隐藏层, 8的倍数, 一般取64, 128, 256, 512, 768等
          "dropout": 0.5,                    # 随机失活, 概率， 0-1
          "activate_mid": "tanh",            # 中间激活函数, 非线性变幻, 提升逼近能力, 选择"relu","tanh"或"sigmoid"
          "activate_end": "softmax",         # 结束激活函数, 即最后一层的激活函数, 如cls激活函数, ner激活函数
          "use_onehot": True,                # label是否使用独热编码
          "use_crf": False,                  # 是否使用CRF(条件随机场), task="sl"(序列标注任务)任务
          "loss": None,                      # 损失函数, 真实值与实际预测的差值损失, 最优化的方向, "categorical_crossentropy"
          "metrics": "accuracy",             # 评估指标, 保存更好模型的评价标准, 一般选择loss, acc或f1等
          "optimizer": "Adam",               # 优化器, 可选["Adam", "Radam", "RAdam,Lookahead"]
          "optimizer_extend":[
              "gradient_accumulation",
              "piecewise_linear_lr",
              "layer_adaptation",
              "lazy_optimization",
              "]weight_decay",
              "lookahead"],                  #   优化器拓展, ["gradient_accumulation", "piecewise_linear_lr", "layer_adaptation",
                                             #              "lazy_optimization","weight_decay", "lookahead"]
          },
"train": {"learning_rate": 1e-3,             # 学习率, 必调参数, 对训练影响较大, word2vec一般设置1e-3, bert设置5e-5或2e-5
          "decay_rate": 0.999,               # 学习率衰减系数, 即乘法, lr = lr * rate
          "decay_step": 1000,                # 学习率每step步衰减, 每N个step衰减一次
          "batch_size": 32,                  # 批处理尺寸, 设置过小会造成收敛困难、陷入局部最小值或震荡, 设置过大会造成泛化能力降低
          "early_stop": 6,                   # 早停, N个轮次(epcoh)评估指标(metrics)不增长就停止训练
          "epochs": 20,                      # 训练最大轮次, 即最多训练N轮
          "label": None,                     # 类别数, auto无需定义, 如果定义则是强制指定
          "is_training": True,               # 是否训练, 用以区分训练train或预测predict, 用它判断后确定加不加载优化器optimizer
           },
"save": {
         # "path_hyper_parameters": None,    # 超参数文件地址
         "path_model_dir": None,   # 模型目录, loss降低则保存的依据, save_best_only=True, save_weights_only=True
         "path_model_info": None,            # 模型所有超参数, 保存在model_info.json
         "path_fineture": None,              # 微调后embedding文件地址, 例如字向量、词向量、bert向量等
          },
"data": {"train_data": None,                 # 训练数据
         "val_data": None                    # 验证数据
         },
}


class Config:
    def __init__(self, hyper_parameters: Dict={}):
        """
        Init of hyper_parameters and build_embed.
        Args:
            hyper_parameters: hyper_parameters of all, which contains "sharing", "embed", "graph", "train", "save" and "data".
        Returns:
            None
        """
        # 各种超参数, 设置默认超参数
        self.hyper_parameters = self.get_hyper_parameters_default()
        # 只更新传入的key-value
        for k in hyper_parameters.keys():
            self.hyper_parameters[k].update(hyper_parameters.get(k, {}))
        self.params_sharing = self.hyper_parameters.get("sharing", {})
        self.params_embed = self.hyper_parameters.get("embed", {})
        self.params_graph = self.hyper_parameters.get("graph", {})
        self.params_train = self.hyper_parameters.get("train", {})
        self.params_save = self.hyper_parameters.get("save", {})
        self.params_data = self.hyper_parameters.get("data", {})
        # params of sharing
        self.gpu_memory_fraction = self.params_sharing.get("gpu_memory_fraction", 0.60)
        self.embed_type = self.params_sharing.get("embed_type", "RANDOM")
        self.token_type = self.params_sharing.get("token_type", "CHAR")
        self.task = self.params_sharing.get("task", None)
        self.length_max = self.params_sharing.get("length_max", None)
        self.vocab_size = self.params_sharing.get("vocab_size", None)
        self.embed_size = self.params_sharing.get("embed_size", None)
        self.trainable = self.params_sharing.get("trainable", True)
        # params of embed
        self.layer_idx = self.params_embed.get("layer_idx", [])
        self.path_embed = self.params_embed.get("path_embed", None)
        self.merge_type = self.params_embed.get("merge_type", "concat")
        self.length_first = self.params_embed.get("length_first", None)
        self.length_second = self.params_embed.get("length_second", None)
        self.xlnet_embed = self.params_embed.get("xlnet_embed", {})
        self.attention_type =  self.params_embed.get("attention_type", "bi")
        self.memory_len =  self.params_embed.get("memory_len", 128)
        self.target_len =  self.params_embed.get("target_len", 128)
        # params of graph
        self.filters_size = self.params_graph.get("filters_size", [3, 4, 5])
        self.filters_num = self.params_graph.get("filters_num", 300)
        self.rnn_type = self.params_graph.get("rnn_type", None)
        self.rnn_unit = self.params_graph.get("rnn_unit", 256)
        self.dropout = self.params_graph.get("dropout", 0.5)
        self.activate_mid = self.params_graph.get("activate_mid", "tanh")
        self.activate_end = self.params_graph.get("activate_end", "softmax")
        self.use_onehot = self.params_graph.get("use_onehot", True)
        self.use_crf = self.params_graph.get("use_crf", False)
        self.loss = self.params_graph.get("loss", "categorical_crossentropy" if self.use_onehot
                                                  else "sparse_categorical_crossentropy")
        self.metrics = self.params_graph.get("metrics", "accuracy")
        self.optimizer = self.params_graph.get("optimizer", "Adam").upper()
        self.optimizer_extend = self.params_graph.get("optimizer_extend", [])
        # params of train
        self.learning_rate = self.params_train.get("learning_rate", 5e-5)
        self.decay_rate = self.params_train.get("decay_rate", 0.999)
        self.decay_step = self.params_train.get("decay_step", 32000)
        self.early_stop = self.params_train.get("early_stop", 6)
        self.batch_size = self.params_train.get("batch_size", 32)
        self.epochs = self.params_train.get("epochs", 20)
        self.label = self.params_train.get("label", None)
        self.is_training = self.params_train.get("is_training", True)
        # params of save
        self.path_model_dir = self.params_save.get("path_model_dir", path_model_dir)
        # self.path_model_info = self.params_save.get("path_model_info", None)
        self.path_fineture = self.params_save.get("path_fineture", None)
        # params of data
        self.train_data = self.params_data.get("train_data", None)
        self.val_data = self.params_data.get("val_data", None)
        # 特殊符号
        self.token_dict = {PAD: 0, UNK: 1,
                           CLS: 2, SEP: 3,
                           BOS: 4, EOS: 5,
                           MASK: 6, WC: 7
                           }
        # 递归创建模型保存目录
        if not self.path_model_dir: self.path_model_dir = path_model_dir
        if not os.path.exists(self.path_model_dir):
            os.makedirs(self.path_model_dir)

    def get_hyper_parameters_default(self) -> Dict:
        """
        Get hyper_parameters of default.
        Args:
            None
        Returns:
            Dict
        """

        return hyper_parameters_default

