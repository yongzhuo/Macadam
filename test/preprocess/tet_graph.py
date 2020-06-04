# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/4/30 20:11
# @author  : Mo
# @function:


# 适配linux
import pathlib
import sys
import os
project_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent)
print(project_path)
# 地址
from macadam.conf.path_config import path_root, path_tc_baidu_qa_2019, path_tc_thucnews
from macadam.base.utils import txt_write, txt_read, load_json, save_json
from macadam.tc.t03_textcnn import TextCNNGraph as Graph
from macadam.base.preprocess import ListPrerocessXY
from macadam.base.embedding import BertEmbedding
from macadam.conf.logger_config import logger
# 计算时间
import time
from macadam import keras, K, L, M, O


# 模型目录
path_model_dir = os.path.join(path_root, "data", "model", "text_cnn_2020")
# 语料地址
path_model = os.path.join(path_model_dir, 'model.h5')
# 超参数保存地址
path_hyper_parameters = os.path.join(path_model_dir, 'hyper_parameters.json')
# embedding微调保存地址
path_fineture = os.path.join(path_model_dir, "embedding_trainable.h5")


if not os.path.exists(path_model_dir):
    os.mkdir(path_model_dir)

 # 训练/验证数据地址
# path_train = os.path.join(path_tc_thucnews, "train.json")
# path_dev = os.path.join(path_tc_thucnews, "dev.json")
path_train = os.path.join(path_tc_baidu_qa_2019, "train.json")
path_dev = os.path.join(path_tc_baidu_qa_2019, "dev.json")



def train(hyper_parameters=None, use_onehot=False, rate=1):
    """
        训练函数
    :param hyper_parameters: json, 超参数
    :param rate: 比率, 抽出rate比率语料取训练
    :return: None
    """

    # 删除先前存在的模型\embedding微调模型等
    time_start = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_KERAS"] = "1"
    path_embed = "D:/soft_install/dataset/bert-model/chinese_L-12_H-768_A-12"
    path_check_point = path_embed + "/bert_model.ckpt"
    path_config = path_embed + "/bert_config.json"
    path_vocab = path_embed + "/vocab.txt"
    length_max = 128

    params = {"embed": {"path_embed": path_embed,
                        "layer_idx": [-2],
                        },
              "sharing": {"length_max": length_max,
                          "embed_size": 768
                          },
              "graph": {"loss": "categorical_crossentropy" if use_onehot else "sparse_categorical_crossentropy",  # 损失函数
                        },
              "save": {
                  "path_model": path_model_dir,  # 模型目录, loss降低则保存的依据, save_best_only=True, save_weights_only=True
                  "path_hyper_parameters": os.path.join(path_model_dir, "hyper_parameters.json"),  # 超参数文件地址
                  "path_fineture": os.path.join(path_model_dir, "embedding.json"),  # 微调后embedding文件地址, 例如字向量、词向量、bert向量等
              },
              }
    bert_embed = BertEmbedding(params)
    bert_embed.build_embedding(path_checkpoint=path_check_point,
                               path_config=path_config,
                               path_vocab=path_vocab)

    graph = Graph(params)

    # 训练/验证数据读取, 每行一个json格式, example: {"x":{"text":"你是谁", "texts2":["你是谁呀", "是不是"]}, "y":"YES"}
    train_data = txt_read(path_train)
    dev_data = txt_read(path_dev)
    # 只有ListPrerocessXY才支持rate(data), 训练比率
    len_train_rate = int(len(train_data) * rate)
    len_dev_rate = int(len(dev_data) * rate)
    train_data = train_data[:len_train_rate]
    dev_data = dev_data[:len_dev_rate]
    pxy = ListPrerocessXY(embedding=bert_embed, data=train_data, path_dir=path_model_dir,
                          length_max=length_max, use_onehot=use_onehot, embed_type="BERT", task="TC")
    from macadam.base.preprocess import ListGenerator as generator_xy
    logger.info("强制使用序列最大长度为{0}, 即文本最大截断或padding长度".format(length_max))
    # 更新最大序列长度, 类别数
    graph.length_max = pxy.length_max
    graph.label = len(pxy.l2i)
    graph.embed_size = bert_embed.embed_size

    # shape = bert_embed.output
    graph.build_model(inputs=bert_embed.model.inputs, outputs=bert_embed.model.output)
    graph.create_compile()
    # 训练
    graph.fit(pxy, generator_xy, train_data, dev_data=dev_data)
    print("耗时:" + str(time.time()-time_start))


if __name__=="__main__":
    train()
    mm = 0
