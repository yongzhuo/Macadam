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
from macadam.conf.path_config import path_tc_baidu_qa_2019, path_tc_thucnews, path_root
from macadam.base.utils import txt_read, txt_write, load_json, save_json
from macadam.tc import ModelPredict
import json


if __name__ == '__main__':
    # init model
    network = "SelfAttention_2020" # "TextCNN" # "FineTune" # "RCNN"
    path_dir = os.path.join(path_root, "data", "model", network)
    mp = ModelPredict(path_dir)
    # 训练/验证数据地址
    path_train = os.path.join(path_tc_baidu_qa_2019, "train.json")
    path_dev = os.path.join(path_tc_baidu_qa_2019, "dev.json")

    # sample
    texts = [{"text": "五彩斑斓的黑",
              "texts2": []}]
    res = mp.predict(texts)
    print(res)
    # evulate
    datas_dev = txt_read(path_dev)
    datas_dev = [json.loads(dd.strip()) for dd in datas_dev[0:64]]
    metrics, report = mp.evaluate(datas_dev)
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


