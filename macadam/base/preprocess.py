# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/4/23 21:17
# @author  : Mo
# @function: preprocess of sentence and label, 1. auto, shuffle-size, sequence-length-max,


from macadam.conf.constant_params import UNK, PAD, MASK, CLS, SEP, PAD, UNK, BOS, EOS, WC, SL, TC, RE
from macadam.base.utils import save_json, load_json, dict_sort, padding_sequences
from macadam.conf.constant_params import EMBEDDING_TYPE
from macadam.conf.path_config import path_model_dir
from macadam.conf.logger_config import logger
from macadam import keras, K

from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import is_string

from typing import Iterable, List, Dict, Any
from collections import Counter, OrderedDict
from tqdm import tqdm
import numpy as np
# import threading
import random
import json
import os


__all__ = ["Tokenizer4Macadam",
           "TokenizerVector",
           "ListPrerocessXY",
           "ListGenerator",
           "FilePrerocessXY",
           "FileGenerator",
           ]


class Tokenizer4Macadam(Tokenizer):
    def __init__(self, token_dict, do_lower_case=False, *args, **kwargs):
        super().__init__(token_dict, do_lower_case=do_lower_case, *args, **kwargs)

    def truncate_sequence_multi(self, length_max: int, sequences: List[Any], pop_index: int = -2):
        """
        truncate sequence of multi, 均衡裁剪List[List]数据, 每一条平均
        Args:
            first_sequence: List, first input of sentence when in single-task, pair-task or multi-task, eg. ["macadam", "英文", "什么", "意思"]
            second_sequence: List, second inputs of sentence, eg. ["macadam", "什么", "意思"] 
            max_length: int, max length of the whole sequence, eg. 512
            pop_index: int, index of start pop, eg. -2, -1
        Returns:
            None
        """
        while True:
            len_total = [len(sequences[i]) for i in range(len(sequences))]
            len_sum = sum(len_total)
            len_max = max(len_total)
            idx = len_total.index(len_max)
            if len_sum <= length_max:
                break
            sequences[idx].pop(pop_index)

    def encode_multi(self, first_text: str, second_texts: List = None, limit_lengths: List = None, length_max: int = None,
                     first_length: int = None, second_length: int = None, is_multi: bool = True):
        """
        bert encode of multi-text-input, 处理多输入(segment_ids标注两个还是多个[0,1]), 同时第二句子可能会截断[CLS]/[SEP]等信息
        Args:
            first_text: Any, first input of sentence when in single-task, pair-task or multi-task, eg. "macadam英文什么意思"
            second_texts: List, second inputs of sentence, eg. ["macadam?", "啥macadam?", "macadam什么意思"] 
            limit_lengths: List, limit lengths of each in texts(second inputs), eg.[36, 36,128]
            length_max: int, max length of the whole sequence, eg. 512
            first_length: int, max length of first_text, eg. 128
            second_length: int, max length of the whole sequence of texts(second inputs), eg. 128
            is_multi: bool, either sign sentence in texts with multi or not
        Returns:
            input of bert-like model
        """

        # split charcter, 分字
        if is_string(first_text):
            first_tokens = self.tokenize(first_text)
        else:
            first_tokens = first_text
        # token to id and pad, 多则截断|少则padding
        first_token_ids = self.tokens_to_ids(first_tokens)
        if first_length:
            first_token_ids = first_token_ids[:first_length]
            first_token_ids.extend([self._token_pad_id] * (first_length - len(first_token_ids)))
        # segment, like [0,0,0,0,0,1,1,1,1], 区分句子
        first_segment_ids = [0] * len(first_tokens)
        # second segments, 第一句以后的句子
        second_token_ids = []
        second_segment_ids = []
        # if texts exist, 如果存在第二句话或者是往上, 避免为空的情况
        if second_texts:
            len_texts = len(second_texts)
            for i in range(len_texts):
                text = second_texts[i]
                if not text:
                    tokens = None
                elif is_string(text):
                    idx = int(bool(self._token_start))
                    tokens = self.tokenize(text)[idx:]
                else:
                    tokens = text
                if tokens:
                    # token to id, 字符转数字
                    token_ids = self.tokens_to_ids(tokens)
                    if limit_lengths and limit_lengths[i]:
                        token_ids = token_ids[:limit_lengths[i]]
                        token_ids.extend([self._token_pad_id] * (limit_lengths[i] - len(token_ids)))
                    # pull segment_id, 句子区分, 第一句为0, 第二句子s是否都标为1
                    if is_multi:
                        id_sent = 1 if i%2==0 else 0
                    else:
                        id_sent = 1
                    segment_ids = [id_sent] * len(token_ids)
                    second_token_ids.extend(token_ids)
                    second_segment_ids.extend(segment_ids)
            # limit second texts, 限制除了第一个句子外的所有句子的长度
            if second_length:
                second_token_ids = second_token_ids[:second_length]
                second_segment_ids = second_segment_ids[:second_length]
            # add all, 所有句子相加
            first_token_ids.extend(second_token_ids)
            first_segment_ids.extend(second_segment_ids)
        # limit all, 限制所有句子的长度
        if length_max:
            first_token_ids = first_token_ids[:length_max]
            first_segment_ids = first_segment_ids[:length_max]

        return first_token_ids, first_segment_ids

    def encode_average(self, first_text: Any, second_texts: Any = None, limit_lengths: List = None, length_max: int = None,
                             first_length: int = None, second_length: int = None, is_multi: bool = True):
        """
        bert encode of multi-text-input, 均衡截断
        Args:
            first_text: Any, first input of sentence when in single-task, pair-task or multi-task, eg. "macadam英文什么意思"
            second_texts: Any, second inputs of sentence, eg. ["macadam?", "啥macadam?", "macadam什么意思"] 
            limit_lengths: List, limit lengths of each in texts(second inputs), eg.[36, 36,128]
            length_max: int, max length of the whole sequence, eg. 512
            first_length: int, max length of first_text, eg. 128
            second_length: int, max length of the whole sequence of texts(second inputs), eg. 128
            is_multi: bool, either sign sentence in texts with multi or not
        Returns:
            input of bert-like model
        """
        length_max = 512 if not length_max else length_max
        # split charcter, 分字
        if is_string(first_text):
            first_tokens = [self.tokenize(first_text)]
        else:
            first_tokens = [[self._token_start] + first_text + [self._token_end]]

        # if texts exist, 如果存在第二句话或者是往上, 避免为空的情况
        if second_texts:
            len_texts = len(second_texts)
            for i in range(len_texts):
                text = second_texts[i]
                if not text:
                    tokens = None
                elif is_string(text):
                    idx = int(bool(self._token_start))
                    tokens = self.tokenize(text)[idx:]
                else:
                    tokens = text + [self._token_end]
                first_tokens.append(tokens)
        # 均衡截断
        if length_max:
            self.truncate_sequence_multi(length_max, first_tokens, -2)
        # token to id and pad, 多则截断|少则padding
        first_token_ids = self.tokens_to_ids(first_tokens[0])
        if first_length:
            first_token_ids = first_token_ids[:first_length]
            first_token_ids.extend([self._token_pad_id] * (first_length - len(first_token_ids)))
        # segment, like [0,0,0,0,0,1,1,1,1], 区分句子
        first_segment_ids = [0] * len(first_token_ids)
        # second segments, 第一句以后的句子
        second_token_ids = []
        second_segment_ids = []
        # if texts exist, 如果存在第二句话或者是往上, 避免为空的情况
        if second_texts:
            len_seconds = len(first_tokens) - 1
            for i in range(len_seconds):
                # token to id, 字符转数字
                token_ids = self.tokens_to_ids(first_tokens[i+1])
                if limit_lengths and limit_lengths[i]:
                    token_ids = token_ids[:limit_lengths[i]]
                    token_ids.extend([self._token_pad_id] * (limit_lengths[i] - len(token_ids)))
                # pull segment_id, 句子区分, 第一句为0, 第二句子s是否都标为1
                if is_multi:
                    id_sent = 1 if i%2==0 else 0
                else:
                    id_sent = 1
                segment_ids = [id_sent] * len(token_ids)
                second_token_ids.extend(token_ids)
                second_segment_ids.extend(segment_ids)
            # limit second texts, 限制除了第一个句子外的所有句子的长度
            if second_length:
                second_token_ids = second_token_ids[:second_length]
                second_segment_ids = second_segment_ids[:second_length]
            # add all, 所有句子相加
            first_token_ids.extend(second_token_ids)
            first_segment_ids.extend(second_segment_ids)

        return first_token_ids, first_segment_ids

    def encode_mix(self, first_text: Any, second_texts: Any = None, limit_lengths: List = None, length_max: int = None,
                         first_length: int = None, second_length: int = None, is_multi: bool = True):
        """
        bert encode of multi-text-input, 均衡截断(混合输入LATTICE-LSTM-BATCH模式情况, 即List[List]情况)
        Args:
            first_text: Any, first input of sentence when in single-task, pair-task or multi-task, eg. "macadam英文什么意思"
            second_texts: Any, second inputs of sentence, eg. ["macadam?", "啥macadam?", "macadam什么意思"] 
            limit_lengths: List, limit lengths of each in texts(second inputs), eg.[36, 36,128]
            length_max: int, max length of the whole sequence, eg. 512
            first_length: int, max length of first_text, eg. 128
            second_length: int, max length of the whole sequence of texts(second inputs), eg. 128
            is_multi: bool, either sign sentence in texts with multi or not
        Returns:
            input of bert-like model
        """
        # split charcter, 分字
        first_tokens = [first_text]
        # if texts exist, 如果存在第二句话或者是往上, 避免为空的情况
        if second_texts:
            len_texts = len(second_texts)
            for i in range(len_texts):
                tokens = second_texts[i]
                first_tokens.append(tokens)
        # 均衡截断
        if length_max:
            self.truncate_sequence_multi(length_max, first_tokens, -2)
        # token to id and pad, 多则截断|少则padding
        first_token_ids = []
        for fts in first_tokens[0]:
            first_token_ids.append(self.tokens_to_ids(fts))
        # 截断最大长度
        if first_length:
            first_token_ids = first_token_ids[:first_length]
            first_token_ids.extend([self.token_to_id(WC)] * (first_length - len(first_token_ids)))
        # segment, like [0,0,0,0,0,1,1,1,1], 区分句子
        first_segment_ids = [0] * len(first_token_ids)
        # second segments, 第一句以后的句子
        second_token_ids = []
        second_segment_ids = []
        # if texts exist, 如果存在第二句话或者是往上, 避免为空的情况
        if second_texts:
            len_seconds = len(first_tokens) - 1
            for i in range(len_seconds):
                # token to id, 字符转数字
                token_ids = []
                for ftsi in first_tokens[i+1]:
                    token_ids.append(self.tokens_to_ids(ftsi))
                if limit_lengths and limit_lengths[i]:
                    token_ids = token_ids[:limit_lengths[i]]
                    token_ids.extend([self.token_to_id(WC)] * (limit_lengths[i] - len(token_ids)))
                # pull segment_id, 句子区分, 第一句为0, 第二句子s是否都标为1
                if is_multi:
                    id_sent = 1 if i%2==0 else 0
                else:
                    id_sent = 1
                segment_ids = [id_sent] * len(token_ids)
                second_token_ids.extend(token_ids)
                second_segment_ids.extend(segment_ids)
            # limit second texts, 限制除了第一个句子外的所有句子的长度
            if second_length:
                second_token_ids = second_token_ids[:second_length]
                second_segment_ids = second_segment_ids[:second_length]
            # add all, 所有句子相加
            first_token_ids.extend(second_token_ids)
            first_segment_ids.extend(second_segment_ids)

        return first_token_ids, first_segment_ids


class ListGenerator(Iterable):
    def __init__(self, data: List, preprocess_xy, shuffle: bool=True, buffer_size: int=3200,
                 batch_size: int=32, len_data: int=None):
        self.preprocess_xy = preprocess_xy
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.len_data = len_data
        self.shuffle = shuffle
        self.data = data

    def __iter__(self):
        """
        iter of train data and preprocess of batch_size
        """
        # yield
        if self.shuffle:
            data_yield = self.generator()
        else:
            data_yield = iter(self.data)
        # batch_x, batch_y, preprocess
        batch_x_idx, batch_y_idx = [], []
        for line in data_yield: # 对x,y进行数据预处理
            line_json = json.loads(line)
            x_id = self.preprocess_xy.preprocess_x(line_json.get("x", {}))
            y_id = self.preprocess_xy.preprocess_y(line_json.get("y", {}))
            batch_x_idx.append(x_id)
            batch_y_idx.append(y_id)
            # yiel when full, 批尺寸满了则处理
            if len(batch_y_idx) == self.batch_size:
                # 转化成 [array(), array]形式
                batch_x_idx_array = []
                batch_y_idx_array = np.array(batch_y_idx)
                # "LATTICE-LSTM-BATCH"的情况单独出来, 即MIX-Embedding得情况
                if self.preprocess_xy.embed_type in ["MIX"]:
                    x_1 = np.array([x[0][0] for x in batch_x_idx])
                    x_2 = np.array([x[1][0] for x in batch_x_idx])
                    batch_x_idx_array = [x_1, x_2]
                    yield batch_x_idx_array, batch_y_idx_array
                elif self.preprocess_xy.embed_type in ["RANDOM", "WORD"]:
                    batch_x_idx_array = np.array(batch_x_idx)
                    yield batch_x_idx_array, batch_y_idx_array
                else:
                    batch_x_idx = np.array(batch_x_idx)
                    for i in range(len(batch_x_idx[0])):
                        bxii_array = np.array([bxi[i] for bxi in batch_x_idx])
                        batch_x_idx_array.append(bxii_array)
                    yield batch_x_idx_array, batch_y_idx_array
                batch_x_idx, batch_y_idx = [], []
        # 最后一轮不足batch_size部分
        if batch_y_idx:
            batch_x_idx_array = []
            batch_y_idx_array = np.array(batch_y_idx)
            # "LATTICE-LSTM-BATCH"的情况单独出来
            if self.preprocess_xy.embed_type in ["MIX"]:
                x_1 = np.array([x[0][0] for x in batch_x_idx])
                x_2 = np.array([x[1][0] for x in batch_x_idx])
                batch_x_idx_array = [x_1, x_2]
                yield batch_x_idx_array, batch_y_idx_array
            elif self.preprocess_xy.embed_type in ["RANDOM", "WORD"]:
                batch_x_idx_array = np.array(batch_x_idx)
                yield batch_x_idx_array, batch_y_idx_array
            else:
                batch_x_idx = np.array(batch_x_idx)
                for i in range(len(batch_x_idx[0])):
                    bxii_array = np.array([bxi[i] for bxi in batch_x_idx])
                    batch_x_idx_array.append(bxii_array)
                yield batch_x_idx_array, batch_y_idx_array

    def __len__(self):
        """
        step of one epcoh
        """
        return max(1, self.len_data//self.batch_size)

    def generator(self):
        """
        generate with shuffle of buffer_size
        """
        stacks, flag = [], False
        for line in self.data:
            stacks.append(line)
            if flag:  # buffer满了以后随机选择一行进行pop
                i = random.randint(0, self.buffer_size-1)
                yield stacks.pop(i)
            elif len(stacks) == self.buffer_size:
                flag = True
        while stacks: # 最后不足buffer的情况
            i = random.randint(0, len(stacks)-1)
            yield stacks.pop(i)

    def forfit(self):
        """
        repeat when fit
        """
        while True:
            for d in self.__iter__():
                yield d


class ListPrerocessXY:
    def __init__(self, embedding, data,
                 length_max: int = None,
                 path_dir: str = path_model_dir,
                 encoding: str = "utf-8",
                 use_onehot: bool = True,
                 embed_type: str = "BERT",
                 task: str = "TC",
                 y_start: str = "O",
                 y_end: str = "O"):
        self.path_vocab_x = os.path.join(path_dir, "vocab_x.txt")
        self.path_vocab_y = os.path.join(path_dir, "vocab_y.txt")
        self.l2i, self.i2l = {}, {}
        self.length_max = length_max
        self.use_onehot = use_onehot
        self.embed_type = embed_type.upper()
        self.embedding = embedding
        self.encoding = encoding
        self.path_dir = path_dir
        self.data = data
        self.task = task.upper()
        self.y_start = y_start
        self.y_end = y_end
        # auto自动模式下自己加载
        if os.environ.get("MACADAM_LEVEL") == "AUTO":
            self.init_params(self.data)

    def analysis_max_length(self, data: List, rate: float = 0.95, length_bert_max:int=512):
        """
        analysis max length of data, 分析最大序列的文本长度
        Args:
            data: List, train data of all, eg. [{"x":{"text":"你", "texts2":["是", "是不是"]}, "y":"YES"}]
            rate: float, covge rate of all datas
            length_bert_max: int, max length of bert-like model's sequence length 
        Returns:
            None
        """
        if self.length_max is None:
            len_sents = []
            for xy in tqdm(data, desc="analysis max length of sentence"):
                xy_json = json.loads(xy.strip())
                x = xy_json.get("x")
                first_text = x.get("text")
                second_texts = x.get("texts2")
                for st in second_texts:
                    first_text += st
                len_sents.append(len(first_text))
            self.length_max = min(sorted(len_sents)[int(rate * len(len_sents))] + 2, length_bert_max)
            logger.info("analysis max length of sentence is {0}".format(self.length_max))

    def analysis_len_data(self, data: str=None):
        """
        统计文本长度, 给定一个数据
        """
        if data:
            self.data = data
        self.len_data = len(self.data)
        return self.len_data

    def build_vocab_y(self, data: List):
        """
        创建列别标签字典等(l2i, i2l), create dict of label
        Args:
            data: List, train data of all, eg. [{"x":{"text":"你", "texts2":["是", "是不是"]}, "y":"YES"}]
        Returns:
            None
        """
        # 统计类别标签, count label
        ys = []
        for xy in tqdm(data, desc="build dict of l2i"):
            xy_json = json.loads(xy.strip())
            y = xy_json.get("y")
            if type(y) == list:
                for yi in y:
                    if yi not in ys:
                        ys.append(yi)
            else:
                if y not in ys:
                    ys.append(y)
        # 创建字典, create dict
        for ysc in ys:
            self.l2i[ysc] = len(self.l2i)
            self.i2l[len(self.l2i)-1] = ysc
        # ner任务的[CLS], [SEP]; 或者是"O"
        if self.task == SL and self.embed_type in EMBEDDING_TYPE:
            if self.y_start not in self.l2i:
                self.l2i[self.y_start] = len(self.l2i)
                self.i2l[len(self.l2i) - 1] = self.y_start
            if self.y_end not in self.l2i:
                self.l2i[self.y_end] = len(self.l2i)
                self.i2l[len(self.l2i) - 1] = self.y_end
        logger.info("build vocab of l2i is {0}".format(self.l2i))
        # # 保存label信息等
        # save_json([self.l2i, self.i2l], self.path_vocab_y, encoding=self.encoding)

    def preprocess_x(self, line_json: Dict, limit_lengths: List=None, use_seconds: bool = True, is_multi: bool = True):
        """
        pre-process with x(sequence) 
        Args:
            line_json: Dict, original inputs of network, eg. {"x":{"text":"你", "texts2":["是", "是不是"]}
            limit_lengths: List, limit lengths of each in texts(second inputs), eg.[36, 36]
            use_seconds: bool, either use [SEP] separate texts2 or not, eg.True
            is_multi: bool, either sign sentence in texts with multi or not, eg. True      
        Returns:
            inputs of bert-like model
        """
        text = line_json.get("text")
        texts2 = line_json.get("texts2", None)
        idxs = self.embedding.sent2idx(text=text, second_text=texts2, limit_lengths=limit_lengths,
                                       use_seconds=use_seconds, is_multi=is_multi)
        if self.embed_type in ["WORD", "RANDOM"]: # and self.task == TC:
            idxs = idxs[0]
        # # LATTICE-LSTM-BATCH单独划分出来
        # if self.embed_type != "MIX":
        #     idxs = padding_sequences(sequences=[idxs] if type(idxs[0])==int else idxs,
        #                              length_max=self.length_max, padding=0)
        # else:
        #     idxs = [idxs[0][0], idxs[1][0]]
        return idxs

    def preprocess_y(self, y):
        """
        pre-process with y(label) 
        Args:
            y: List or str, input of label, eg. "YES"
            use_onehot: bool, whether use onehot encoding label or not, eg.True     
        Returns:
            inputs of label
        """
        if type(y) == list:
            if self.task == SL:
                [y] = padding_sequences(sequences=[y], length_max=self.length_max,
                                        padding="O", task=SL,
                                        padding_start=self.y_start, padding_end=self.y_end)
                # 补全y的cls和sep
                y = [self.y_start] + y[:self.length_max-2] + [self.y_end]
            label = [self.l2i[yi] for yi in y]
        else:
            label = self.l2i[y]

        if self.use_onehot:
            from tensorflow.python.keras.utils import to_categorical
            label = to_categorical(label, num_classes=len(self.l2i))

        return label

    def init_params(self, data: List, rate: float = 0.95):
        """
        analysis max length of data and build dict of label, 分析最大序列的文本长度以及创建类别标签字典
        Args:
            data: List, train data of all, eg. [{"x":{"text":"你", "texts2":["是", "是不是"]}, "y":"YES"}]
            rate: float, covge rate of all datas
            length_bert_max: int, max length of bert-like model's sequence length 
        Returns:
            None
        """
        self.analysis_max_length(data, rate)
        self.build_vocab_y(data)


class FileGenerator(Iterable):
    def __init__(self, path: str,
                 preprocess_xy,
                 shuffle: bool = True,
                 buffer_size: int = 32000,
                 batch_size: int = 32,
                 len_data: int = None,
                 encoding: str = "utf-8"):
        self.path = path
        self.preprocess_xy = preprocess_xy
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.len_data = len_data
        self.encoding = encoding
        # self.lock = threading.Lock()

    def __iter__(self):
        """
        iter of train data and preprocess of batch_size
        """
        # yield
        data_yield = self.generator()
        # batch_x, batch_y, preprocess
        batch_x_idx, batch_y_idx = [], []
        for line in data_yield: # 对x,y进行数据预处理
            line_json = json.loads(line)
            x_id = self.preprocess_xy.preprocess_x(line_json.get("x"))
            y_id = self.preprocess_xy.preprocess_y(line_json.get("y"))
            batch_x_idx.append(x_id)
            batch_y_idx.append(y_id)
            # yiel when full, 批尺寸满了则处理
            if len(batch_y_idx) == self.batch_size:
                # 转化成 [array(), array]形式
                batch_x_idx_array = []
                batch_y_idx_array = np.array(batch_y_idx)
                # "LATTICE-LSTM-BATCH"的情况单独出来, 即MIX-Embedding得情况
                if self.preprocess_xy.embed_type in ["MIX"]:
                    x_1 = np.array([x[0][0] for x in batch_x_idx])
                    x_2 = np.array([x[1][0] for x in batch_x_idx])
                    batch_x_idx_array = [x_1, x_2]
                    yield batch_x_idx_array, batch_y_idx_array
                elif self.preprocess_xy.embed_type in ["RANDOM", "WORD"]:
                    batch_x_idx_array = np.array(batch_x_idx)
                    yield batch_x_idx_array, batch_y_idx_array
                else:
                    batch_x_idx = np.array(batch_x_idx)
                    for i in range(len(batch_x_idx[0])):
                        bxii_array = np.array([bxi[i] for bxi in batch_x_idx])
                        batch_x_idx_array.append(bxii_array)
                    yield batch_x_idx_array, batch_y_idx_array
                batch_x_idx, batch_y_idx = [], []
                # 最后一轮不足batch_size部分
        if batch_y_idx:
            batch_x_idx_array = []
            batch_y_idx_array = np.array(batch_y_idx)
            # "LATTICE-LSTM-BATCH"的情况单独出来
            if self.preprocess_xy.embed_type in ["MIX"]:
                x_1 = np.array([x[0][0] for x in batch_x_idx])
                x_2 = np.array([x[1][0] for x in batch_x_idx])
                batch_x_idx_array = [x_1, x_2]
                yield batch_x_idx_array, batch_y_idx_array
            elif self.preprocess_xy.embed_type in ["RANDOM", "WORD"]:
                batch_x_idx_array = np.array(batch_x_idx)
                yield batch_x_idx_array, batch_y_idx_array
            else:
                batch_x_idx = np.array(batch_x_idx)
                for i in range(len(batch_x_idx[0])):
                    bxii_array = np.array([bxi[i] for bxi in batch_x_idx])
                    batch_x_idx_array.append(bxii_array)
                yield batch_x_idx_array, batch_y_idx_array

    def __len__(self):
        """
        step of one epcoh
        """
        return max(1, self.len_data//self.batch_size)

    def generator(self):
        """
        generate with shuffle of buffer_size, whether shuffle or not
        """
        stacks, flag = [], False
        with open(self.path, "r", encoding=self.encoding) as fr:
            if self.shuffle:
                for line in fr:
                    stacks.append(line)
                    if flag:  # buffer满了以后随机选择一行进行pop
                        i = random.randint(0, self.buffer_size-1)
                        yield stacks.pop(i)
                    elif len(stacks) == self.buffer_size:
                        flag = True
                while stacks: # 最后不足buffer的情况
                    i = random.randint(0, len(stacks)-1)
                    yield stacks.pop(i)
            else:
                for line in fr:
                    yield line
            fr.close()

    def forfit(self):
        """
        repeat when fit
        """
        # with self.lock:
        while True:
            for d in self.__iter__():
                yield d


class FilePrerocessXY:
    def __init__(self, embedding, path,
                 length_max: int = None,
                 path_dir: str = path_model_dir,
                 encoding: str = "utf-8",
                 use_onehot: bool = True,
                 embed_type: str = "BERT",
                 task: str = "TC",
                 y_start: str = "O",
                 y_end: str = "O"):
        self.l2i, self.i2l = {}, {}
        self.length_max = length_max
        self.use_onehot = use_onehot
        self.embed_type = embed_type
        self.embedding = embedding
        self.encoding = encoding
        self.path_dir = path_dir
        self.path = path
        self.task = task
        self.y_start = y_start
        self.y_end = y_end
        self.len_data = None
        # auto自动模式下自己加载
        if os.environ.get("MACADAM_LEVEL") == "AUTO":
            self.init_params(self.path)

    def analysis_max_length(self, path: str=None, rate: float=0.95, length_bert_max:int=512):
        """
        analysis max length of data, 分析最大序列的文本长度, 未统计cls,sep等
        Args:
            path: str, train data file, eg. "/home/data/textclassification/baidu_qa_2019/train.json"
            rate: float, covge rate of all datas
            length_bert_max: int, max length of bert-like model's sequence length 
        Returns:
            None
        """

        if path and os.path.exists(path):
            self.path = path
        # 如果没强制指定最大长度
        if self.length_max is None:
            len_sents = []
            with open(self.path, "r", encoding=self.encoding) as fr:
                for xy in tqdm(fr, desc="analysis max length of sentence"):
                    xy_json = json.loads(xy.strip())
                    x = xy_json.get("x")
                    first_text = x.get("text")
                    second_texts = x.get("texts2")
                    for st in second_texts:
                        first_text += st
                    len_sents.append(len(first_text))
            # 取得覆盖rate(0.95)语料的长度
            self.length_max = min(sorted(len_sents)[int(rate * len(len_sents))] + 2, length_bert_max)
            logger.info("analysis max length of sentence is {0}".format(self.length_max))

    def analysis_len_data(self, path: str=None):
        """
        analysis sample of path, 统计文本长度
        Args:
            path: str, train data file, eg. "/home/data/textclassification/baidu_qa_2019/train.json"
        Returns:
            None
        """
        self.len_data = 0
        if path and os.path.exists(path):
            self.path = path
        with open(self.path, "r", encoding=self.encoding) as fr:
            for _ in tqdm(fr, desc="build dict of l2i"):
                self.len_data += 1
            fr.close()
        return self.len_data

    def build_vocab_y(self, path: str=None):
        """
        创建列别标签字典等(l2i, i2l), create dict of label
        Args:
            path: str, train data file, eg. "/home/data/textclassification/baidu_qa_2019/train.json"
        Returns:
            None
        """
        if path and os.path.exists(path):
            self.path = path
        # 统计类别标签, count label
        ys_counter = []
        ys = []
        with open(self.path, "r", encoding=self.encoding) as fr:
            for xy in tqdm(fr, desc="build dict of l2i"):
                xy_json = json.loads(xy.strip())
                y = xy_json.get("y")
                if type(y) == list:
                    for yi in y:
                        if yi not in ys:
                            ys.append(yi)
                    ys_counter += y
                else:
                    if y not in ys:
                        ys.append(y)
                    ys_counter.append(y)
            fr.close()
        # 类别统计
        ys_counter_dict = dict(Counter(ys_counter))
        ys_counter_dict_sort = dict_sort(ys_counter_dict)
        logger.info(json.dumps(ys_counter_dict_sort, ensure_ascii=False, indent=4))
        # 创建字典, create dict
        for ysc in ys:
            self.l2i[ysc] = len(self.l2i)
            self.i2l[len(self.l2i) - 1] = ysc
        # ner任务的[CLS], [SEP]; 或者是"O"
        if self.task == SL and self.embed_type in EMBEDDING_TYPE:
            if self.y_start not in self.l2i:
                self.l2i[self.y_start] = len(self.l2i)
                self.i2l[len(self.l2i) - 1] = self.y_start
            if self.y_end not in self.l2i:
                self.l2i[self.y_end] = len(self.l2i)
                self.i2l[len(self.l2i) - 1] = self.y_end
        logger.info("build vocab of l2i is {0}".format(self.l2i))

    def preprocess_x(self, line_json: Dict,
                     limit_lengths: List=None,
                     use_seconds: bool = True,
                     is_multi: bool = True):
        """
        pre-process with x(sequence) 
        Args:
            line_json: Dict, original inputs of network, eg. {"x":{"text":"你", "texts2":["是", "是不是"]}
            limit_lengths: List, limit lengths of each in texts(second inputs), eg.[36, 36]
            use_seconds: bool, either use [SEP] separate texts2 or not, eg.True
            is_multi: bool, either sign sentence in texts with multi or not, eg. True      
        Returns:
            inputs of bert-like model
        """
        text = line_json.get("text")
        texts2 = line_json.get("texts2", None)
        idxs = self.embedding.sent2idx(text=text, second_text=texts2, limit_lengths=limit_lengths,
                                       use_seconds=use_seconds, is_multi=is_multi)
        if self.embed_type in ["WORD", "RANDOM"]:  # and self.task == TC:
            idxs = idxs[0]
        return idxs

    def preprocess_y(self, y):
        """
        pre-process with y(label) 
        Args:
            y: List or str, input of label, eg. "YES"
            use_onehot: bool, whether use onehot encoding label or not, eg.True     
        Returns:
            inputs of label
        """
        if type(y) == list:
            if self.task == SL:
                [y] = padding_sequences(sequences=[y], length_max=self.length_max,
                                        padding="O", task=SL,
                                        padding_start=self.y_start, padding_end=self.y_end)
                # 补全y的cls和sep
                y = [self.y_start] + y[:self.length_max-2] + [self.y_end]
            label = [self.l2i[yi] for yi in y]
        else:
            label = self.l2i[y]

        if self.use_onehot:
            from tensorflow.python.keras.utils import to_categorical
            label = to_categorical(label, num_classes=len(self.l2i))

        return label

    def init_params(self, path: str, rate: float = 0.95):
        """
        analysis max length of data and build dict of label, 分析最大序列的文本长度以及创建类别标签字典
        Args:
            path: str, train data file, eg. "/home/data/textclassification/baidu_qa_2019/train.json"
            rate: float, covge rate of all datas
            length_bert_max: int, max length of bert-like model's sequence length 
        Returns:
            None
        """
        self.analysis_max_length(path, rate)
        self.build_vocab_y(path)


if __name__ == "__main__":
    mm = 0

