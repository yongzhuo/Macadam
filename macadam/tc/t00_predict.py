# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/5/8 21:38
# @author  : Mo
# @function: class of model predict


# 适配linux
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(path_root)
## cpu-gpu与tf.keras
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_KERAS"] = "1"

from macadam.base.utils import padding_sequences, metrics_report
from macadam.base.layers import custom_objects_macadam
from macadam.base.embedding import embedding_map
from macadam.base.utils import load_json
from macadam import keras, K, L, M, O
from collections import OrderedDict
from typing import List, Dict
from tqdm import tqdm
import numpy as np


class ModelPredict():
    def __init__(self, path_dir):
        """
        init
        """
        self.path_model_info = os.path.join(path_dir, "macadam.info")
        self.path_model_h5 = os.path.join(path_dir, "macadam.h5")
        self.path_dir = path_dir
        os.environ["MACADAM_LEVEL"] = "PREDICT"
        self.load_tokenizer()
        self.load_model()

    def load_model(self):
        """
        load model of keras of h5 which include graph-node and custom_objects        
        """
        self.model = M.load_model(self.path_model_h5, compile=False)

    def load_tokenizer(self):
        """
        load model_info of model, hyper_parameters/label2index/index2label/vocab_dict
        """
        self.model_info = load_json(self.path_model_info)
        self.l2i = self.model_info.get("label", {}).get("l2i", {})
        self.i2l = self.model_info.get("label", {}).get("i2l", {})
        hyper_parameters = self.model_info.get("hyper_parameters", {})
        embed_type = hyper_parameters.get("sharing", {}).get("embed_type", "bert").upper()
        token2idx = self.model_info.get("vocab", {}).get("token2idx", {})
        Embedding = embedding_map.get(embed_type)
        self.embedd = Embedding(hyper_parameters)

        self.embedd.build_tokenizer_from_dict(token2idx)
        self.length_max = hyper_parameters.get("sharing", {}).get("length_max", 512)
        self.batch_size = hyper_parameters.get("sharing", {}).get("batch_size", 32)

    def preprocess_x(self, line_json, limit_lengths: List=None,
                     use_seconds: bool = True,
                     is_multi: bool = True):
        """
        data preprocess of encode
        Args:
            line_json: Dict, input, eg. {"text": "macadam是什么", "texts2": ["macadam是一个python工具包]} 
            limit_lengths: List, max length of each enum in texts2, eg.[128]
            use_seconds: bool, either use [SEP] separate texts2 or not, eg.True
            is_multi: bool, either sign texts2 with [0-1; 0] or not, eg.True
        Returns:
            res: List[Dict]
        """
        text = line_json.get("text")
        texts2 = line_json.get("texts2", None)
        idxs = self.embedd.sent2idx(text=text, second_text=texts2, limit_lengths=limit_lengths,
                                       use_seconds=use_seconds, is_multi=is_multi)
        # sequence接受的是List[List], WORD/RANDOM嵌入时候需要加List
        # idxs = padding_sequences(sequences=[idxs] if type(idxs[0])==int else idxs,
        #                          length_max=self.length_max, padding=0)
        return idxs

    def predict(self, texts: List[Dict],
                use_sort: bool = True) -> List[Dict]:
        """
        model predict
        Args:
            texts: input of List<dict>, eg. [{"text": "macadam是什么", "texts2": ["macadam是一个python工具包]}] 
        Returns:
            res: List[Dict]
        """
        # embedding编码, bert encode
        xs = []
        for text_i in texts:
            text_i_x = self.preprocess_x(text_i)
            xs.append(text_i_x)
        # numpy处理, numpy.array
        xs_array = []
        idxs_np = np.array(xs)
        for i in range(len(idxs_np[0])):
            idxs_array = np.array([inxi[i] for inxi in idxs_np])
            xs_array.append(idxs_array)
        # 模型预测, model predict
        xs_prob = self.model.predict(xs_array)
        # 后处理, post preprocess
        res = []
        for x_prob in xs_prob:
            x_dict = {}
            for i in range(len(self.i2l)):
                x_dict[self.i2l[str(i)]] = x_prob[i]
            res.append(x_dict)

        if use_sort:
            res_sort = [sorted(p.items(), key=lambda x: x[1], reverse=True) for p in res]
            res_sort_order = [OrderedDict(rs) for rs in res_sort]
            res = [{k: v for k, v in x.items()} for x in res_sort_order]
        return res

    def evaluate(self, texts: List[Dict]):
        """
        evaluate of corpus, 数据集验证/打印报告
        Args:
            texts: input of List<dict>, eg. [{"text": "macadam是什么", "texts2": ["macadam是一个python工具包]}] 
        Returns:
            res: List[Dict]
        """
        labels_true = []
        labels_pred = []
        texts_batch = []
        # tqdm显示进度
        for i in tqdm(range(len(texts))):
            line = texts[i]
            texts_batch.append(line)
            if len(texts_batch)==self.batch_size:
                # true_y
                labels_true_batch = [tsb.get("y", []) for tsb in texts_batch]
                # pred_y
                texts_batch_x = [tsb.get("x", {}) for tsb in texts_batch]
                labels_predict_batch = self.predict(texts_batch_x)
                # 处理y_true大于length_max的情况
                for i in range(len(labels_predict_batch)):
                    labels_pred += [list(labels_predict_batch[i].keys())[0]]
                    labels_true += [labels_true_batch[i]]
                texts_batch = []
        # storage, Less than batch_size, 剩余不足批处理尺寸的
        if texts_batch:
            # true_y
            labels_true_batch = [tsb.get("y", []) for tsb in texts_batch]
            # pred_y
            texts_batch_x = [tsb.get("x", {}) for tsb in texts_batch]
            labels_predict_batch = self.predict(texts_batch_x)
            # 处理y_true大于length_max的情况
            for i in range(len(labels_predict_batch)):
                labels_pred += [list(labels_predict_batch[i].keys())[0]]
                labels_true += [labels_true_batch[i]]
        # 获取评估指标/报告打印
        mertics, report = metrics_report(y_true=labels_true, y_pred=labels_pred)
        return mertics, report

if __name__ == '__main__':
    from macadam.conf.path_config import path_root, path_tc_baidu_qa_2019
    from macadam.base.utils import txt_write, txt_read
    import json

    # 模型目录与加载
    path_dir = os.path.join(path_root, "data", "model", "TextCNN_2020")
    mp = ModelPredict(path_dir)
    texts = [{"text": "五彩斑斓的黑",
              "texts2": []}]
    # 预测
    res = mp.predict(texts)
    print(res)
    # path_train = os.path.join(path_tc_thucnews, "train.json")
    # path_dev = os.path.join(path_tc_thucnews, "dev.json")
    path_train = os.path.join(path_tc_baidu_qa_2019, "train.json")
    path_dev = os.path.join(path_tc_baidu_qa_2019, "dev.json")
    datas_dev = txt_read(path_dev)
    datas_dev = [json.loads(dd.strip()) for dd in datas_dev]

    # evaluate
    metrics, report = mp.evaluate(datas_dev)
    print(json.dumps(metrics, ensure_ascii=False, indent=4))
    print(report)

    # ccks-2020-task-1
    # path_root_kg_2020 = "D:/soft_install/dataset/game/CCKS/ccks_kg_2020/ccks_7_1_competition_data"
    # path_train = "验证集"
    # path_all = os.path.join(path_root_kg_2020, path_train, "entity_validation.txt")
    # questions = txt_read(path_all)
    #
    # res_last = []
    # for ques in questions:
    #     text = ques.strip()
    #     texts = {"text": text,
    #              "texts2": []}
    #     res = mp.predict([texts])
    #     # print(res)
    #     # res_0 = res[0]
    #     res_sort = [sorted(p.items(), key=lambda x: x[1], reverse=True) for p in res]
    #     label = res_sort[0][0][0]
    #     line = ques + "\t" + label + "\n"
    #     res_last.append(line)
    # txt_write(res_last, "entity_validation_14.txt")

    mm = 0






