# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/5/8 21:12
# @author  : Mo
# @function: class of model predict


# 适配linux
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(path_root)
# cpu-gpu与tf.keras
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_KERAS"] = "1"
# macadam
from macadam.conf.path_config import path_root, path_ner_people_1998, path_ner_clue_2020
from macadam.base.utils import txt_read, txt_write, load_json, save_json
from macadam.sl import ModelPredict
import json
import os


if __name__ == '__main__':
    # 模型加载, load model
    network = "CRF" # "CRF", "Bi-LSTM-CRF", "Bi-LSTM-LAN", "CNN-LSTM", "DGCNN", "LATTICE-LSTM-BATCH"
    path_dir = os.path.join(path_root, "data", "model", network)
    mp = ModelPredict(path_dir)
    # 训练/验证数据地址
    path_train = os.path.join(path_ner_people_1998, "train.json")
    path_dev = os.path.join(path_ner_people_1998, "dev.json")
    path_tet = os.path.join(path_ner_people_1998, "text.json")
    # path_train = os.path.join(path_ner_clue_2020, "ner_clue_2020.train")
    # path_dev = os.path.join(path_ner_clue_2020, "ner_clue_2020.dev")
    # sample
    texts = [{"text": "你的一腔热情，别人只道是狼心狗肺"
                      "一切往事，皆为序章"
                      "never say never"
                      "那就这样了吧"
                      "再见，北京",
              "texts2": []}
             ]
    res = mp.predict(texts)
    print(res)
    # evaluate
    datas_dev = txt_read(path_tet)
    print("evaluate开始！")
    datas_dev = [json.loads(dd.strip()) for dd in datas_dev]
    metrics, report = mp.evaluate(datas_dev)
    print("evaluate结束！")
    print(json.dumps(metrics, ensure_ascii=False, indent=4))
    print(report)
    # input
    while True:
        print("请输入 text1:")
        text = input()
        texts = {"text": text,
                 "texts2": []}
        res = mp.predict([texts])
        print(res)

    mm = 0

