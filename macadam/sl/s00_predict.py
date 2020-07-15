# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/5/8 21:38
# @author  : Mo
# @function: class of model predict of sequence-labeling


# 适配linux
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(path_root)
## cpu-gpu与tf.keras
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_KERAS"] = "1"
# macadam
from macadam.conf.path_config import path_root, path_ner_people_1998, path_ner_clue_2020
from macadam.base.utils import txt_read, txt_write, load_json, save_json
from macadam.base.utils import padding_sequences, metrics_report
from macadam.base.layers import custom_objects_macadam
from macadam.conf.constant_params import SL, CLS, SEP
from macadam.base.embedding import embedding_map
from macadam.base.utils import load_json
from macadam import keras, K, L, M, O
from collections import OrderedDict
from typing import List, Dict
from tqdm import tqdm
import numpy as np
import json
import os


# keras.utils.get_custom_objects().update(custom_objects_macadam)
# custom_objects = keras.utils.get_custom_objects()


class ModelPredict:
    def __init__(self, path_dir: str):
        """
        init, 序列标注任务模型预测类
        """
        self.path_model_info = os.path.join(path_dir, "macadam.info")
        self.path_model_h5 = os.path.join(path_dir, "macadam.h5")
        self.path_dir = path_dir
        # os.environ["MACADAM_LEVEL"] = "PREDICT"
        self.load_tokenizer()
        self.load_model()

    def load_model(self):
        """
        load model of keras of h5 which include graph-node and custom_objects        
        """
        self.model = M.load_model(self.path_model_h5, compile=False)

    def load_tokenizer(self):
        """
        load model_info of model, hyper_parameters/label2index/index2label/token2idx
        """
        # 从字典里边读取数据
        self.model_info = load_json(self.path_model_info)
        hyper_parameters = self.model_info.get("hyper_parameters", {})
        self.embed_type = hyper_parameters.get("sharing", {}).get("embed_type", "bert").upper()
        self.length_max = hyper_parameters.get("sharing", {}).get("length_max", 512)
        self.batch_size = hyper_parameters.get("sharing", {}).get("batch_size", 32)
        self.token2idx = self.model_info.get("vocab", {}).get("token2idx", {})
        self.use_crf = hyper_parameters.get("graph", {}).get("use_crf", True)
        self.l2i = self.model_info.get("label", {}).get("l2i", {})
        self.i2l = self.model_info.get("label", {}).get("i2l", {})
        # 初始化embedding
        Embedding = embedding_map.get(self.embed_type)
        self.embedd = Embedding(hyper_parameters)
        # 使用CRF就需要trans(状态转移矩阵)和维特比解码
        if self.use_crf:
            self.trans = self.model_info.get(SL, {}).get("trans", [])

        # 字典构建Tokenizer, MIX-embedding单独提取出来
        if self.embed_type not in ["MIX"]:
            self.embedd.build_tokenizer_from_dict(self.token2idx)
        else:
            self.embedd.build_tokenizer_from_dict(self.token2idx)

    def preprocess_x(self, line_json: Dict, limit_lengths: List=None,
                     use_seconds: bool = True, is_multi: bool = True) -> List[List]:
        """
        data pre-process of encode, 数据预处理
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
        # idxs = padding_sequences(sequences=[idxs] if type(idxs[0])==int else idxs,
        #                          length_max=self.length_max, padding=0)
        return idxs

    def viterbi_decode(self, nodes: np.array, trans: np.array) -> np.array:
        """
        viterbi decode of CRF, 维特比解码, Viterbi算法求最优路径
        code from url: https://github.com/bojone/bert4keras
        author       : bojone
        Args:
            nodes: np.array, shape=[seq_len, num_labels], output of model predict
            trans: np.array, shape=[num_labels, num_labels], state transition matrix
        Returns:
            res: np.array, label of sequence
        """
        labels = np.arange(len(self.l2i)).reshape((1, -1))
        scores = nodes[0].reshape((-1, 1))
        scores[1:] -= np.inf  # 第一个标签必然是0
        paths = labels
        for l in range(1, len(nodes)):
            M = scores + trans + nodes[l].reshape((1, -1))
            idxs = M.argmax(0)
            scores = M.max(0).reshape((-1, 1))
            path_idxs = paths[:, idxs]
            paths = np.concatenate([path_idxs, labels], 0)
        return paths[:, scores[0].argmax()]

    def predict(self, texts: List[Dict], use_sort: bool = True) -> List[Dict]:
        """
        model predict， 批处理模型预测
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
        # "LATTICE-LSTM-BATCH"的情况单独出来, 即MIX-Embedding的情况
        if self.embed_type in ["MIX"]:
            x_1 = np.array([x[0][0] for x in xs])
            x_2 = np.array([x[1][0] for x in xs])
            xs_array = [x_1, x_2]
        else:
            for i in range(len(xs[0])):
                idxs_array = np.array([inxi[i] for inxi in xs])
                xs_array.append(idxs_array)
        # 模型预测, model predict
        labels = self.model.predict(xs_array)
        labels_argmax = [l.argmax(-1) for l in labels]
        if self.use_crf:
            trans = np.array(self.trans)
            labels_argmax = [self.viterbi_decode(label, trans) for label in labels]
        # 后处理, post-processing
        labels_zh = []
        for i in range(len(labels_argmax)):
            # 返回文本的真实长度
            len_text_i = min(len(texts[i].get("text")) + 2, self.length_max)
            la = labels_argmax[i]
            label_zh = []
            for lai in la:
                label_zh.append(self.i2l[str(lai)])
            # 随机初始化的就没有[CLS], [SEP]
            # if self.embed_type in ["RANDOM", "WORD2VEC"]:
            #     labels_zh.append(label_zh[0:len_text_i - 2])
            # else:
            labels_zh.append(label_zh[1:len_text_i-1])
        return labels_zh

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
        # predict of batch_size, 批处理预测
        texts_batch = []
        # tqdm显示进度
        for i in tqdm(range(len(texts))):
            line = texts[i]
            texts_batch.append(line)
            if len(texts_batch) == self.batch_size:
                # true_y
                labels_true_batch = [tsb.get("y", []) for tsb in texts_batch]
                # pred_y
                texts_batch_x = [tsb.get("x", {}) for tsb in texts_batch]
                labels_predict_batch = self.predict(texts_batch_x)
                # 处理y_true大于length_max的情况
                for i in range(len(labels_predict_batch)):
                    labels_pred += labels_predict_batch[i]
                    labels_true += labels_true_batch[i][:len(labels_predict_batch[i])]
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
                labels_pred += labels_predict_batch[i]
                labels_true += labels_true_batch[i][:len(labels_predict_batch[i])]
        # 获取评估指标/报告打印
        mertics, report = metrics_report(y_true=labels_true, y_pred=labels_pred)
        return mertics, report

    def explain_pos(self, words0, tag1):
        res = []
        ws = ""
        start_pos = 0
        end_pos = 0
        types = ""
        sentence = ""
        for i in range(len(tag1)):
            if tag1[i].startswith("S-"):
                ws += words0[i]
                start_pos = len(sentence)
                sentence += words0[i]
                end_pos = len(sentence) - 1
                types = tag1[i][2:]
                res.append([ws, start_pos, end_pos, types])
                ws = ""
                types = ""
            if tag1[i].startswith("B-"):
                if len(ws) > 0:
                    res.append([ws, start_pos, end_pos, types])
                    ws = ""
                    types = ""
                if len(ws) == 0:
                    ws += words0[i]
                    start_pos = len(sentence)
                    sentence += words0[i]
                    end_pos = len(sentence) - 1
                    types = tag1[i][2:]

            elif tag1[i].startswith("M-"):
                if len(ws) > 0 and types == tag1[i][2:]:
                    ws += words0[i]
                    sentence += words0[i]
                    end_pos = len(sentence) - 1
                elif len(ws) > 0 and types != tag1[i][2:]:
                    res.append([ws, start_pos, end_pos, types])
                    ws = ""
                    types = ""
                if len(ws) == 0:
                    ws += words0[i]
                    start_pos = len(sentence)
                    sentence += words0[i]
                    end_pos = len(sentence) - 1
                    types = tag1[i][2:]

            elif tag1[i].startswith("E-"):
                if len(ws) > 0 and types == tag1[i][2:]:
                    ws += words0[i]
                    sentence += words0[i]
                    end_pos = len(sentence) - 1
                    res.append([ws, start_pos, end_pos, types])
                    ws = ""
                    types = ""
                if len(ws) > 0 and types != tag1[i][2:]:
                    res.append([ws, start_pos, end_pos, types])
                    ws = ""
                    ws += words0[i]
                    start_pos = len(sentence)
                    sentence += words0[i]
                    end_pos = len(sentence) - 1
                    types = tag1[i][2:]
                    res.append([ws, start_pos, end_pos, types])
                    ws = ""
                    types = ""
            elif tag1[i] == "O":
                sentence += words0[i]
            if i == len(tag1) - 1 and len(ws) > 0:
                res.append([ws, start_pos, end_pos, types])
                ws = ""
                types = ""

        res1 = []
        for s in res:
            s1 = {}
            s1["word"] = s[0]
            s1["start_pos"] = s[1]
            s1["end_pos"] = s[2]
            s1["entity_type"] = s[3]
            res1.append(s1)
        res2 = {}
        textss = "".join(words0)
        res2["text"] = textss
        res2["label_results"] = res1
        return res2


if __name__ == "__main__":
    from macadam.conf.path_config import path_root

    # 模型目录与初始化
    path_dir = os.path.join(path_root, "data", "model", "CRF")
    mp = ModelPredict(path_dir)
    # streamer = ThreadedStreamer(predict_function=mp.predict,
    #                             batch_size=32,
    #                             max_latency=0.01,)
    # streamer = Streamer(predict_function_or_model=mp.predict,
    #                     batch_size=32,
    #                     max_latency=0.01,
    #                     worker_num=1,
    #                     cuda_devices=(0),
    #                     mp_start_method="fork")
    # 训练/验证数据地址
    path_train = os.path.join(path_ner_people_1998, "train.json")
    path_dev = os.path.join(path_ner_people_1998, "dev.json")
    # path_train = os.path.join(path_ner_clue_2020, "ner_clue_2020.train")
    # path_dev = os.path.join(path_ner_clue_2020, "ner_clue_2020.dev")

    # sample
    texts = [{"text": "你的一腔热情，别人只道是狼心狗肺"
                      "一切往事，皆为序章"
                      "never say never"
                      "那就这样了吧"
                      "再见，北京"
                      ,
              "texts2": []}]
    res = mp.predict(texts)
    print(res)
    # evaluate
    datas_dev = txt_read(path_dev)
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