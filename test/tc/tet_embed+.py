# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/5/9 21:07
# @author  : Mo
# @function: test embedding(bert4keras, not Embedding) + finetune(write freedom)


# 适配linux
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(path_root)
# 地址, tf.keras
from macadam.base.preprocess import ListPrerocessXY, ListGenerator
from macadam.base.embedding import embedding_map
from macadam.conf.path_config import path_root, path_tc_baidu_qa_2019
from macadam.conf.logger_config import logger
from macadam.tc.t00_map import graph_map
from macadam.base.utils import txt_write
from macadam.base.utils import txt_read
from macadam import keras, K, L, M, O, C
# 计算时间
import time


# cpu-gpu与tf.keras
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_KERAS"] = "1"

# bert-embedding地址, 必传
path_embed = "D:/soft_install/dataset/bert-model/chinese_L-12_H-768_A-12"
path_checkpoint = path_embed + "/bert_model.ckpt"
path_config = path_embed + "/bert_config.json"
path_vocab = path_embed + "/vocab.txt"

# 训练/验证数据地址
path_train = os.path.join(path_tc_baidu_qa_2019, "train.json")
path_dev = os.path.join(path_tc_baidu_qa_2019, "dev.json")
network="FastText"
embed="Bert"
is_length_max=False
layer_idx=[-1]
length_max=128
embed_size=768
learning_rate=1e-5
batch_size=32
epochs=20
early_stop=3
decay_rate=0.999
decay_step=1000
rate=1.0

path_model_dir = os.path.join(path_root, "data", "model", f"{network}_20200502")

# 获取embed和graph的类
Embedding = embedding_map[embed.upper()]
Graph = graph_map[network.upper()]

# 删除先前存在的模型/embedding微调模型等
time_start = time.time()
# bert-embedding等初始化
params = {"embed": {"path_embed": path_embed,
                    "layer_idx": layer_idx,
                    },
          "sharing": {"length_max": length_max,
                      "embed_size": embed_size
                      },
          "train": {"learning_rate": learning_rate,  # 学习率, 必调参数, 对训练影响较大, word2vec一般设置1e-3, bert设置5e-5或2e-5
                    "decay_rate": decay_rate,  # 学习率衰减系数, 即乘法, lr = lr * rate
                    "decay_step": decay_step,  # 学习率每step步衰减, 每N个step衰减一次
                    "batch_size": batch_size,  # 批处理尺寸, 设置过小会造成收敛困难、陷入局部最小值或震荡, 设置过大会造成泛化能力降低
                    "early_stop": early_stop,  # 早停, N个轮次(epcoh)评估指标(metrics)不增长就停止训练
                    "epochs": epochs,  # 训练最大轮次, 即最多训练N轮
                    },
          "save": {
              "path_model_dir": path_model_dir,  # 模型目录, loss降低则保存的依据, save_best_only=True, save_weights_only=True
              "path_model_info": os.path.join(path_model_dir, "model_info.json"),  # 超参数文件地址
          },
          }

embed = Embedding(params)
embed.build_embedding(path_checkpoint=path_checkpoint,
                      path_config=path_config,
                      path_vocab=path_vocab)

# 训练/验证数据读取, 每行一个json格式, example: {"x":{"text":"你是谁", "texts2":["你是谁呀", "是不是"]}, "y":"YES"}
train_data = txt_read(path_train)
dev_data = txt_read(path_dev)

len_train_rate = int(len(train_data) * rate)
len_dev_rate = int(len(dev_data) * rate)

train_data = train_data[:len_train_rate]
dev_data = dev_data[:len_dev_rate]


logger.info("训练/验证语料读取完成")
# 数据预处理类初始化
preprocess_xy = ListPrerocessXY(embed, train_data, path_dir=path_model_dir, length_max=length_max)

x = L.Lambda(lambda x: x[:, 0], name="Token-CLS")(embed.model.output)

# 最后就是softmax
outputs = L.Dense(len(preprocess_xy.l2i), activation="softmax",
                       kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02))(x)
model = M.Model(embed.model.input, outputs)
model.summary(132)

model.compile(optimizer=O.Adam(lr=1e-5),
                           loss="categorical_crossentropy",
                           metrics=["accuracy"])

len_train_data = len(train_data)
lg_train = ListGenerator(train_data, preprocess_xy, batch_size=batch_size, len_data=len_train_data)
lg_dev = None
# monitor是早停和保存模型的依据, "loss", "acc", "val_loss", "val_acc"等
monitor = "val_loss"
if dev_data:
    len_dev_data = len(dev_data)
    lg_dev = ListGenerator(dev_data, preprocess_xy, batch_size=batch_size, len_data=len_dev_data)
else:
    monitor = "loss"

call_back = [C.TensorBoard(log_dir=os.path.join(path_model_dir, "logs"), batch_size=batch_size, update_freq='batch'),
             C.EarlyStopping(monitor=monitor, mode="auto", min_delta=1e-9, patience=early_stop),
             C.ModelCheckpoint(monitor=monitor, mode="auto", filepath=os.path.join(path_model_dir, "macadam.h5"),
                               verbose=1, save_best_only=True, save_weights_only=False)]
# 训练模型
model.fit_generator(generator=lg_train.forfit(),
                    steps_per_epoch=lg_train.__len__(),
                    callbacks=call_back,
                    epochs=epochs,
                    validation_data=lg_dev.forfit(),
                    validation_steps=lg_dev.__len__())
