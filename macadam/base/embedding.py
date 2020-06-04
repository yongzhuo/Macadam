# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2020/4/23 21:29
# @author   :Mo
# @function :embeddings of model, base embedding of random, word2vec or bert-like model
# @warning  :MixEmbedding is only used by LATTICE-LSTM-BATCH, MIX嵌入只适用于LATTICE-LSTM-BATCH


from macadam.base.utils import load_word2vec_format, macropodus_cut, get_ngram, padding_sequences, is_total_chinese
from macadam.conf.constant_params import Config, UNK, PAD, MASK, CLS, SEP, PAD, UNK, BOS, EOS, WC
from macadam.base.preprocess import Tokenizer4Macadam as Tokenizer
from macadam.base.layers import NonMaskingLayer, SelfAttention
from bert4keras.models import build_transformer_model
from macadam.conf.logger_config import logger
from macadam import keras, K, L, M, O
from typing import Dict, Any, List
from tqdm import tqdm
import numpy as np
import codecs
import json
import os


__all__ = ["BaseEmbedding",
           "RandomEmbedding",
           "WordEmbedding",
           "BertEmbedding",
           "RoBertaEmbedding",
           "AlBertEmbedding",
           "XlnetEmbedding",
           "NezhaEmbedding",
           "ElectraEmbedding",
           "Gpt2Embedding",
           "MixEmbedding"
           ]


class BaseEmbedding(Config):
    def __init__(self, hyper_parameters: Dict):
        """
        Init of hyper_parameters and build_embed.
        Args:
            hyper_parameters: hyper_parameters of all, which contains "sharing", "embed", "graph", "train", "save" and "data".
        Returns:
            None
        """
        # 各种超参数, 设置默认超参数
        super().__init__(hyper_parameters)
        # bert-like model下最大长度为512, 即bert4keras的max_position
        if self.embed_type not in ["RANDOM", "WORD", "NGRAM"]:
            self.length_max = min(self.length_max, 512)
        # # auto自动模式下自己加载
        # if os.environ.get("MACADAM_LEVEL")=="AUTO":
        #     self.build_embedding()

    def load_embed_model(self):
        """
        Load embedding model, which pre-train with corpus outside.
        Args:
            None
        Returns:
            None
        """
        pass

    def build_tokenizer(self):
        """
        Load tokenizer, form vocab most possible.
        Args:
            None
        Returns:
            None
        """

        pass

    def build_embedding(self):
        """
        build model of embedding, with keras or tensorflow.python.keras.
        Args:
            None
        Returns:
            None
        """
        self.token2idx = {}
        self.idx2token = {}


class RandomEmbedding(BaseEmbedding):
    def __init__(self, hyper_parameters: Dict):
        """
        Init of hyper_parameters and build_embed.
        Args:
            hyper_parameters: hyper_parameters of all, which contains "sharing", "embed", "graph", "train", "save" and "data".
        Returns:
            None
        """
        hyper_parameters["sharing"].update({"embed_type": "RANDOM",})
        self.ngram_ns = hyper_parameters.get("embed", {}).get("ngram_ns", [2, 3])  # ngram信息, 根据语料获取
        super().__init__(hyper_parameters)

    def build_tokenizer(self, path_vocab: str=None):
        """
        build tokenizer from path_vocab. 构建字/词字典(从字-词典/训练语料)
        Args:
            path_vocab: str, path_corpus  or  constant file of "default_character.txt"
        Returns:
            None
        """
        # 避免为空
        path_vocab = "" if not path_vocab else path_vocab
        # constant sign
        self.token2idx = self.token_dict.copy()
        # count = len(self.token2idx) - 1
        # 默认的字符(char)
        if "default_character.txt" in path_vocab:
            with open(file=path_vocab, mode="r", encoding="utf-8") as fd:
                while True:
                    line = fd.readline()
                    if not line:
                        break
                    line = line.strip()
                    if line not in self.token2idx:
                        # count = count + 1
                        self.token2idx[line] = len(self.token2idx)
        # 训练语料的数据
        elif os.path.exists(path_vocab):
            with open(file=path_vocab, mode="r", encoding="utf-8") as fd:
                for fd_line in tqdm(fd):
                    # 处理 {"x":{"text":"", "texts2":[]}, "y":""}格式
                    fd_line_json = json.loads(fd_line.strip())
                    line_x = fd_line_json.get("x", {})
                    line = line_x.get("text")
                    line_texts2 = line_x.get("texts2")
                    if line_texts2:
                        line += "".join(line_texts2)
                    # 数据预处理, 切词等
                    if self.token_type.upper() == "CHAR":
                        text = list(line.replace(" ", "").strip())
                    elif self.token_type.upper() == "WORD":
                        text = macropodus_cut(line)
                    elif self.token_type.upper() == "NGRAM":
                        text = get_ngram(line, ns=self.ngram_ns)
                    else:
                        raise RuntimeError("your input level_type is wrong, it must be 'word', 'char', 'ngram'")
                    # 获取token
                    for line in text:
                        if line not in self.token2idx:
                            self.token2idx[line] = len(self.token2idx)
                            # count = count + 1
                            # self.token2idx[line] = count
        else:
            # raise RuntimeError("your input path_embed is wrong, it must be csv, txt or tsv")
            logger.info("path_corpus is not exists!")
        self.idx2token = {}
        for k, v in self.token2idx.items():
            self.idx2token[v] = k

    def build_tokenizer_from_dict(self, token2idx):
        """
        reader and create tokenizer from dict, 构建字/词字典(从字典dict)
        Args:
            token2idx: dict, vocab of token2idx, eg. {"[PAD]":0, "[MASK]":1}   
        Returns:
            None
        """
        self.vocab_size = len(token2idx)
        self.token2idx = token2idx
        self.tokenizer = Tokenizer(token2idx, do_lower_case=True)

    def select_tokens(self):
        """
        select tokens from token_dict or word2vec, 选择tokens
        Args:
            None
        Returns:
            None
        """
        # 创建词向量矩阵, 存在则用预训练好的, 不存在则随机初始化, [PAD], [UNK], [BOS], [EOS]......
        embedding_matrix = []
        # 第一个是[PAD], zeros全零
        embedding_matrix.append(np.zeros(self.embed_size))
        # 只选择corpus里边的token, 可以防止embedding矩阵OOM
        for _ in tqdm(range(len(self.token2idx)-1)):
            embedding_matrix.append(np.random.uniform(-0.5, 0.5, self.embed_size))
        embedding_matrix = np.array(embedding_matrix)
        return embedding_matrix

    def build_embedding(self, path_checkpoint: str=None,
                          path_config: str=None,
                          path_vocab: str=None,
                          path_corpus: str=None):
        """
        build embedding of word2vec or fasttext, 构建Embedding
        Args:
            path_checkpoint: No use
            path_config: No use
            path_vocab: No use
            path_embed: str, file path of word2vec, eg. "/home/embedding/word2vec/sgns.wiki.word"
            path_corpus: str, file path of train data, eg. "/home/corpus/ner_people_1998/train.json"
        Returns:
            None
        """
        # 训练语料地址(用于创建字/词典)
        if not path_corpus or not os.path.exists(path_corpus):
             path_corpus = self.train_data
        # 创建token2idx
        self.build_tokenizer(path_corpus)
        # 构建Tokenizer
        self.build_tokenizer_from_dict(self.token2idx)
        # 挑选词语, 构建embedding_matrix, 随机uniform初始化
        embedding_matrix = self.select_tokens()
        # embed
        self.vocab_size = len(self.token2idx)
        self.input = L.Input(shape=(self.length_max, ), dtype="int32")
        self.output = L.Embedding(self.vocab_size,
                                  self.embed_size,
                                  input_length=self.length_max,
                                  trainable=self.trainable,
                                  weights=[embedding_matrix],
                                  )(self.input)
        self.model = M.Model(self.input, self.output)

    def cut_and_index(self, text: str):
        """
        cut sentence and index it, 切词并转化为数据number
        Args:
            text: str, original inputs of text, eg. "macadam是什么！"   
        Returns:
            List
        """
        if self.token_type.upper() == "CHAR":
            text = list(text)
        elif self.token_type.upper() == "WORD":
            text = macropodus_cut(text)
        elif self.token_type.upper() == "NGRAM":
            text = get_ngram(text, ns=self.ngram_ns)
        else:
            raise RuntimeError("your input level_type is wrong, "
                               "it must be 'word' or 'char'")
        # # token 转 index(number)
        # text_index = [self.token2idx[ti] if ti in self.token2idx
        #               else self.token2idx[UNK] for ti in text]
        # return text_index
        return text

    def sent2idx(self, text: Any = None, second_text: Any = None, limit_lengths: List = None,
                 use_seconds: bool = True, is_multi: bool = True):
        """
        cut/index/padding sentence, 切词/数字转化/pad 文本
        Args:
            text: Any, first input of sentence when in single-task, pair-task or multi-task, eg. "macadam英文什么意思"
            second_text: Any, second inputs of sentence, eg. ["macadam?", "啥macadam?", "macadam什么意思"] 
            limit_lengths: List, limit lengths of each in texts(second inputs), eg.[36, 36,128]
            use_seconds: bool, whether use class"encode_multi" or not
            is_multi: bool, whether sign sentence in texts with multi or not
        Returns:
            List
        """
        text = self.cut_and_index(text)
        second_text = [self.cut_and_index(st) for st in second_text]
        if use_seconds: # 第二个句子传入多个, 即定义多个[SEP]
            input_id, id_type = self.tokenizer.encode_average(first_text=text,
                                                              second_texts=second_text,
                                                              limit_lengths=limit_lengths,
                                                              length_max=self.length_max,
                                                              first_length=self.length_first,
                                                              second_length=self.length_second,
                                                              is_multi=is_multi)
            # input_mask = [0 if ids == 0 else 1 for ids in input_id]
            idxs = [input_id, id_type]
        else:
            input_id, input_type_id = self.tokenizer.encode(first_text=text,
                                                            second_text=second_text,
                                                            max_length=self.length_max,
                                                            first_length=self.length_first,
                                                            second_length=self.length_second)
            # input_mask = [0 if ids == 0 else 1 for ids in input_id]
            idxs = [input_id, input_type_id]
        idxs = padding_sequences(sequences=[idxs] if type(idxs[0]) == int else idxs, task=self.task,
                                 padding_start=self.token2idx[CLS], length_max=self.length_max,
                                 padding_end=self.token2idx[SEP], padding=self.token2idx[PAD],
                                 )
        return idxs

    def encode(self, text: Any = None, second_text: Any = None, limit_lengths: List = None,
               use_seconds: bool = True, is_multi: bool = True) -> List[List]:
        """
        encode sentence, 文本编码(model.predict)
        Args:
            text: Any, first input of sentence when in single-task, pair-task or multi-task, eg. "macadam英文什么意思"
            second_text: List, second inputs of sentence, eg. ["macadam?", "啥macadam?", "macadam什么意思"] 
            limit_lengths: List, limit lengths of each in texts(second inputs), eg.[36, 36,128]
            use_seconds: bool, whether use class"encode_multi" or not
            is_multi: bool, whether sign sentence in texts with multi or not
        Returns:
            List
        """
        sent_ids = self.sent2idx(text=text, second_text=second_text, is_multi=is_multi,
                                 limit_lengths=limit_lengths, use_seconds=use_seconds,)
        sent_idx_np = np.array([sent_ids[0]])
        res = self.model.predict(sent_idx_np)
        return res


class WordEmbedding(RandomEmbedding):
    def __init__(self, hyper_parameters):
        hyper_parameters["sharing"].update({"embed_size": 300,
                                            "embed_type": "WORD",
                                            })
        super().__init__(hyper_parameters)

    def select_tokens(self):
        """
        select tokens from token_dict or word2vec, 选择tokens
        Args:
            None
        Returns:
            None
        """
        # 加载静态词向量
        self.word_vector = load_word2vec_format(path=self.path_embed, limit=self.limit)
        # 适配调整嵌入向量的维度
        self.embed_size = self.word_vector[list(self.word_vector.keys())[0]].size
        # 创建词向量矩阵, 存在则用预训练好的, 不存在则随机初始化, [PAD], [UNK], [BOS], [EOS]......
        embedding_matrix = []
        # 如果字典较小, 那么就不是只使用corpus语料的词语, 即word2vec里边的token全部使用
        if self.vocab_size >= 320:
            for k, v in self.token2idx.items():
                if k in self.word_vector:
                    v_word2vec = self.word_vector[k]
                else:
                    v_word2vec = np.random.uniform(-0.5, 0.5, self.embed_size)
                embedding_matrix.append(v_word2vec)
        # 只选择corpus里边的token, 可以防止embedding矩阵OOM
        else:
            embedding_matrix.append(np.zeros(self.embed_size))
            self.token2idx = self.token_dict.copy()
            for _ in range(len(self.token2idx)-1):
                embedding_matrix.append(np.random.uniform(-0.5, 0.5, self.embed_size))
            for k, v in self.word_vector.items():
                self.token2idx[k] = len(self.token2idx)
                embedding_matrix.append(self.word_vector[k])
        embedding_matrix = np.array(embedding_matrix)
        return embedding_matrix

    def build_embedding(self, path_checkpoint: str = None, path_config: str = None, path_vocab: str = None,
                        path_embed: str = None, path_corpus: str = None, limit: int=None):
        """
        build embedding of word2vec or fasttext
        Args:
            path_checkpoint: No use
            path_config: No use
            path_vocab: No use
            path_embed: str, file path of word2vec, eg. "/home/embedding/word2vec/sgns.wiki.word"
            path_corpus: str, file path of train data, eg. "/home/corpus/ner_people_1998/train.json"
            limit: int, limit token of vocab, eg. 100
        Returns:
            None
        """
        # 训练语料地址(用于创建字/词典)
        if not path_corpus or not os.path.exists(path_corpus):
             path_corpus = self.train_data
        # 预训练embedding地址
        if path_embed and os.path.exists(path_embed):
            self.path_embed = path_embed
        self.limit = limit

        # 构建词表字典
        self.build_tokenizer(path_corpus)
        # 字典大小
        self.vocab_size = len(self.token2idx)
        # 创建词向量矩阵, 存在则用预训练好的, 不存在则随机初始化, [PAD], [UNK], [BOS], [EOS]......
        embedding_matrix = self.select_tokens()
        self.embed_size = embedding_matrix.shape[1]
        # 筛选后的词典大小
        self.vocab_size = len(self.token2idx)
        # 构建Tokenizer
        self.build_tokenizer_from_dict(self.token2idx)
        # embed
        # embedding_matrix = np.array(embedding_matrix)
        self.input = L.Input(shape=(self.length_max,), dtype="int32")
        self.output = L.Embedding(self.vocab_size,
                                  self.embed_size,
                                  input_length=self.length_max,
                                  weights=[embedding_matrix],
                                  trainable=self.trainable)(self.input)
        self.model = M.Model(self.input, self.output)


class BertEmbedding(BaseEmbedding):
    def __init__(self, hyper_parameters):
        """
        Init of hyper_parameters and build_embed.
        Args:
            hyper_parameters: hyper_parameters of all, which contains "sharing", "embed", "graph", "train", "save" and "data".
        Returns:
            None
        """
        hyper_parameters["sharing"].update({"embed_size": 768,
                                            "token_type": "CHAR",
                                            "embed_type": "BERT",
                                            })
        super().__init__(hyper_parameters)

    def build_config(self, path_config: str=None):
        # reader config of bert
        self.configs = {}
        if path_config is not None:
            self.configs.update(json.load(open(path_config)))

    def build_tokenizer(self, path_vocab: str=None):
        """
        build tokenizer from path_vocab. 构建字典(从预训练好的bert-like模型)
        Args:
            path_vocab: str, path_vocab, eg. "/home/embedding/bert/chinese_L-12_H-768_A-12/vocab.txt"
        Returns:
            None
        """
        # reader and create tokenizer
        if not path_vocab:
            path_vocab = os.path.join(self.path_embed, "vocab.txt")
        self.token2idx = {}
        with codecs.open(path_vocab, "r", "utf-8") as reader:
            for line in reader:
                token = line.strip()
                self.token2idx[token] = len(self.token2idx)
        self.vocab_size = len(self.token2idx)
        self.tokenizer = Tokenizer(self.token2idx, do_lower_case=True)

    def build_tokenizer_from_dict(self, token2idx):
        """
        build tokenizer from token2idx. 构建tokenizer(从字典dict)
        Args:
            token2idx: dict, token2idx, eg. {"[PAD]":0, "UNK":1}
        Returns:
            None
        """
        # reader and create tokenizer
        self.vocab_size = len(token2idx)
        self.token2idx = token2idx
        self.tokenizer = Tokenizer(token2idx, do_lower_case=True)

    def build_merge_layer(self):
        """
        build, get and merge layer. 构建bert/获取层/创建层与层间关系
        Args:
            None
        Returns:
            None
        """
        # 获取实际的layer
        features_layers = [self.output_layers[li] for li in self.layer_idx]
        # 输出layer层的merge方式
        if len(features_layers) == 1:
            # embedding_layer = L.AveragePooling1D()([features_layers[0], features_layers[0]])
            embedding_layer = features_layers[0]
        elif self.merge_type == "concat":
            embedding_layer = L.Concatenate()(features_layers)
        elif self.merge_type == "add":
            embedding_layer = L.Add()(features_layers)
        elif self.merge_type == "multi": # 不作处理, 输出指定所有层
            embedding_layer = L.Lambda(lambda x:x)(features_layers)
        elif self.merge_type == "pool-max":
            # features_layers_2 = L.Concatenate()(features_layers)
            features_layers_max = [L.GlobalMaxPooling1D()(fl) for fl in features_layers]
            embedding_layer = L.Lambda(lambda x:x)(features_layers_max)
        elif self.merge_type == "pool-avg":
            features_layers_max = [L.AveragePooling1D()(fl) for fl in features_layers]
            embedding_layer = L.Lambda(lambda x: x)(features_layers_max)
        else:
            raise RuntimeError("your input merge_type is wrong, it must be 'concat', 'add', 'pool-max' or 'pool-avg'")
        output_layer = NonMaskingLayer()(embedding_layer)
        # 整个模型输入输出
        self.outputs = output_layer
        # self.output = [embedding_layer]
        self.inputs = self.model.inputs # must input
        self.model = M.Model(self.inputs, self.outputs)
        # self.model.summary(132)

    def build_embedding(self, return_keras_model: bool = True, path_checkpoint: str = None, path_config: str = None,
                        path_vocab: str = None, application: str = "encoder", model: str = "bert", **kwargs):
        """
        build embedding of bert, 创建bert嵌入模块
        Args:
            path_checkpoint: str, file path of checkpoint of bert, eg. "/home/embedding/bert/chinese_L-12_H-768_A-12/bert_model.ckpt"
            path_config: str, file path of config of bert, eg. "/home/embedding/bert/chinese_L-12_H-768_A-12/bert_config.json"
            path_vocab:  str, file path of vocab of bert, eg. "/home/embedding/bert/chinese_L-12_H-768_A-12/vocab.txt" 
            path_embed: str, file path of bert, eg. "/home/embedding/bert/chinese_L-12_H-768_A-12"
            path_corpus: str, file path of train data, eg. "/home/corpus/ner_people_1998/train.json"
            limit: int, limit token of vocab, eg. 100
        Returns:
            None
        """
        # 地址
        if not path_checkpoint:
            path_checkpoint = os.path.join(self.path_embed, "bert_model.ckpt")
        if not path_config:
            path_config = os.path.join(self.path_embed, "bert_config.json")
        if not path_vocab:
            path_vocab = os.path.join(self.path_embed, "vocab.txt")
        # 序列最大长度, 不得高于512
        kwargs["sequence_length"] = min(self.length_max, 512)
        # 加载bert等预训练模型
        self.model = build_transformer_model(return_keras_model=return_keras_model,
                                             checkpoint_path=path_checkpoint,
                                             config_path=path_config,
                                             application=application,
                                             model=model,
                                             **kwargs
                                             )
        self.build_config(path_config)
        # 获取bert4keras层输出
        num_hidden_layers = self.configs.get("num_hidden_layers", 12)
        self.output_layers = [self.model.get_layer('Transformer-{0}-FeedForward-Norm'.format(i)).output
                              for i in range(num_hidden_layers)]
        # merge layer of output, 输出层合并处理concat,add,pool-max or pool-avg
        self.build_merge_layer()
        # tokenizer
        self.build_tokenizer(path_vocab)

    def sent2idx(self, text: Any = None, second_text: Any = None, limit_lengths: List = None,
                 use_seconds: bool = True, is_multi: bool = True) -> List[List]:
        """
        cut/index/padding sentence, 切词/数字转化/pad 文本
        Args:
            text: Any, first input of sentence when in single-task, pair-task or multi-task, eg. "macadam英文什么意思"
            second_text: Any, second inputs of sentence, eg. ["macadam?", "啥macadam?", "macadam什么意思"] 
            limit_lengths: List, limit lengths of each in texts(second inputs), eg.[36, 36,128]
            use_seconds: bool, whether use class"encode_multi" or not
            is_multi: bool, whether sign sentence in texts with multi or not
        Returns:
            List
        """
        if use_seconds: # 第二个句子传入多个, 即定义多个[SEP]
            input_id, id_type = self.tokenizer.encode_average(first_text=text,
                                                              second_texts=second_text,
                                                              limit_lengths=limit_lengths,
                                                              length_max=self.length_max,
                                                              first_length=self.length_first,
                                                              second_length=self.length_second,
                                                              is_multi=is_multi)
            # input_mask = [0 if ids == 0 else 1 for ids in input_id]
            idxs = [input_id, id_type]
        else:
            input_id, input_type_id = self.tokenizer.encode(first_text=text,
                                                            second_text=second_text,
                                                            max_length=self.length_max,
                                                            first_length=self.length_first,
                                                            second_length=self.length_second)
            # input_mask = [0 if ids == 0 else 1 for ids in input_id]
            idxs = [input_id, input_type_id]
        idxs = padding_sequences(sequences=[idxs] if type(idxs[0]) == int else idxs,
                                 length_max=self.length_max, padding=self.token2idx[PAD], task=self.task,
                                 padding_start=self.token2idx[CLS], padding_end=self.token2idx[SEP]
                                 )
        return idxs

    def encode(self, text: Any = None, second_text: Any = None, limit_lengths: List = None,
               use_seconds: bool = True, is_multi: bool = True) -> List[List]:
        """
        encode sentence, 文本编码(model.predict)
        Args:
            text: Any, first input of sentence when in single-task, pair-task or multi-task, eg. "macadam英文什么意思"
            second_text: List, second inputs of sentence, eg. ["macadam?", "啥macadam?", "macadam什么意思"] 
            limit_lengths: List, limit lengths of each in texts(second inputs), eg.[36, 36,128]
            use_seconds: bool, whether use class"encode_multi" or not
            is_multi: bool, whether sign sentence in texts with multi or not
        Returns:
            List
        """
        sent_ids = self.sent2idx(text=text, second_text=second_text, is_multi=is_multi,
                                 limit_lengths=limit_lengths, use_seconds=use_seconds,)
        sent_idx_np = [np.array([sent_ids[0]]), np.array([sent_ids[1]])]
        res = self.model.predict(sent_idx_np)
        return res


class RoBertaEmbedding(BertEmbedding):
    def __init__(self, hyper_parameters):
        super().__init__(hyper_parameters)


class AlBertEmbedding(BertEmbedding):
    def __init__(self, hyper_parameters):
        hyper_parameters["sharing"].update({"embed_size": 768,
                                            "token_type": "CHAR",
                                            "embed_type": "ALBERT",
                                            })
        super().__init__(hyper_parameters)

    def build_embedding(self, return_keras_model: bool = True, path_checkpoint: str = None, path_config: str = None,
                        path_vocab: str = None, application: str = "encoder", model: str = "albert", **kwargs):
        # 地址
        if not path_checkpoint:
            path_checkpoint = os.path.join(self.path_embed, "albert_model.ckpt")
        if not path_config:
            path_config = os.path.join(self.path_embed, "albert_config.json")
        if not path_vocab:
            path_vocab = os.path.join(self.path_embed, "vocab.txt")
        # 序列最大长度, 不得高于512
        kwargs["sequence_length"] = min(self.length_max, 512)
        # 加载bert等预训练模型
        self.model = build_transformer_model(return_keras_model=return_keras_model,
                                             max_position_embeddings=self.length_max,
                                             checkpoint_path=path_checkpoint,
                                             config_path=path_config,
                                             application=application,
                                             model=model,
                                             **kwargs
                                             )
        # 获取超参数config.json
        self.build_config(path_config)
        # 获取bert4keras层输出
        num_hidden_layers = self.configs.get("num_hidden_layers", 12)
        output_layer = 'Transformer-FeedForward-Norm'
        self.output_layers = [self.model.get_layer(output_layer).get_output_at(i - 1)
                              for i in range(num_hidden_layers)]
        # merge layer of output, 输出层合并处理concat,add,pool-max or pool-avg
        self.build_merge_layer()
        # tokenizer
        self.build_tokenizer(path_vocab)


class NezhaEmbedding(BertEmbedding):
    def __init__(self, hyper_parameters):
        hyper_parameters["sharing"].update({"embed_size": 768,
                                            "token_type": "CHAR",
                                            "embed_type": "NEZHA",
                                            })
        super().__init__(hyper_parameters)

    def build_embedding(self, path_checkpoint: str = None, path_config: str = None, path_vocab: str = None,
                        application: str = "encoder", model: str = "nezha", return_keras_model: bool = True, **kwargs):
        # 地址
        if not path_checkpoint:
            path_checkpoint = os.path.join(self.path_embed, "bert_model.ckpt")
        if not path_config:
            path_config = os.path.join(self.path_embed, "bert_config.json")
        if not path_vocab:
            path_vocab = os.path.join(self.path_embed, "vocab.txt")
        # 序列最大长度, 不得高于512
        kwargs["sequence_length"] = min(self.length_max, 512)
        # 加载bert等预训练模型
        self.model = build_transformer_model(return_keras_model=return_keras_model,
                                             max_position_embeddings=self.length_max,
                                             checkpoint_path=path_checkpoint,
                                             config_path=path_config,
                                             application=application,
                                             model=model,
                                             **kwargs
                                             )
        # 获取超参数config.json
        self.build_config(path_config)
        # 获取bert4keras层输出
        num_hidden_layers = self.configs.get("num_hidden_layers", 12)
        self.output_layers = [self.model.get_layer('Transformer-{0}-FeedForward-Norm'.format(i)).output
                              for i in range(num_hidden_layers)]
        # merge layer of output, 输出层合并处理concat,add,pool-max or pool-avg
        self.build_merge_layer()
        # tokenizer
        self.build_tokenizer(path_vocab)


class XlnetEmbedding(BertEmbedding):
    def __init__(self, hyper_parameters):
        # 定义死的, 在该模式下
        hyper_parameters["sharing"].update({"embed_size": 768,
                                            "token_type": "CHAR",
                                            "embed_type": "XLNET",
                                            })
        super().__init__(hyper_parameters)

    def build_tokenizer(self, path_vocab: str = None):
        from keras_xlnet import Tokenizer
        if not path_vocab:
            path_vocab = os.path.join(self.path_embed, "spiece.model")
        self.tokenizer = Tokenizer(path_vocab)

    def build_embedding(self, path_checkpoint: str = None,
                        path_config: str = None,
                        path_vocab: str = None,
                        attention_type: str = None,
                        memory_len: int = None,
                        target_len: int = None,
                        **kwargs
                        ):
        from keras_xlnet import load_trained_model_from_checkpoint, set_custom_objects
        from keras_xlnet import ATTENTION_TYPE_BI, ATTENTION_TYPE_UNI
        if not path_checkpoint:
            path_checkpoint = os.path.join(self.path_embed, "xlnet_model.ckpt")
        if not path_config:
            path_config = os.path.join(self.path_embed, "xlnet_config.json")
        if not attention_type:
            attention_type = self.xlnet_embed.get("attention_type", "bi")  # "bi" or "uni"
            self.attention_type = ATTENTION_TYPE_BI if attention_type == "bi" else ATTENTION_TYPE_UNI
        if not memory_len:
            self.memory_len = self.xlnet_embed.get("memory_len", 512)
        if not target_len:
            self.target_len = self.xlnet_embed.get("target_len", 512)
        if not path_vocab:
            path_vocab =os.path.join(self.path_embed,"spiece.model")
        # xlnet加载
        self.model = load_trained_model_from_checkpoint(checkpoint_path=path_checkpoint,
                                                        attention_type=self.attention_type,
                                                        in_train_phase=self.trainable,
                                                        batch_size=self.batch_size,
                                                        config_path=path_config,
                                                        memory_len=self.memory_len,
                                                        target_len=self.target_len,
                                                        mask_index=0)
        # graph, keras.utils.get_custom_objects
        set_custom_objects()
        self.build_config(path_config)
        self.config_model_bert = self.model.get_config()
        # 获取bert4keras层输出
        num_hidden_layers = self.configs.get("n_layer", 12)
        output_layer = "FeedForward-Normal-{0}"
        self.output_layers = [self.model.get_layer(output_layer.format(i+1)).get_output_at(node_index=0)
                              for i in range(num_hidden_layers)]
        # merge layer of output, 输出层合并处理concat,add,pool-max or pool-avg
        self.build_merge_layer()
        # tokenizer
        self.build_tokenizer(path_vocab)

    def sent2idx(self, text: Any=None,
                       second_text: Any=None) -> List[Any]:

        tokens = self.tokenizer.encode(text)
        tokens = tokens + [0] * (self.target_len - len(tokens)) if len(tokens) < self.target_len \
                               else tokens[0:self.target_len]
        token_input = np.expand_dims(np.array(tokens), axis=0)
        segment_input = np.zeros_like(token_input)
        memory_length_input = np.zeros((1, 1))
        masks = [1] * len(tokens) + ([0] * (self.target_len - len(tokens))
                                                   if len(tokens) < self.target_len else [])
        mask_input = np.expand_dims(np.array(masks), axis=0)
        if self.trainable:
            idxs = [token_input, segment_input, memory_length_input, mask_input]
        else:
            idxs = [token_input, segment_input, memory_length_input]
        # padding
        idxs = padding_sequences(sequences=[idxs] if type(idxs[0]) == int else idxs,
                                 length_max=self.length_max, padding=self.token2idx[PAD], task=self.task,
                                 padding_start=self.token2idx[CLS], padding_end=self.token2idx[SEP]
                                )
        return idxs

    def encode(self, text: Any = None,
               second_text: Any = None) -> List[List]:

        sent_ids = self.sent2idx(text=text, second_text=second_text)
        sent_idx_np = [np.array([sent_ids[0]]), np.array([sent_ids[1]])]
        res = self.model.predict(sent_idx_np)
        return res


class ElectraEmbedding(BertEmbedding):
    def __init__(self, hyper_parameters):
        hyper_parameters["sharing"].update({"embed_size": 768,
                                            "token_type": "CHAR",
                                            "embed_type": "ELECTRA",
                                            })
        super().__init__(hyper_parameters)

    def build_embedding(self, path_checkpoint: str = None, path_config: str = None, path_vocab: str = None,
                        return_keras_model: bool = True, application: str = "encoder",
                        model: str = "electra", **kwargs):
        # 地址
        if not path_checkpoint:
            path_checkpoint = os.path.join(self.path_embed, "electra_model.ckpt")
        if not path_config:
            path_config = os.path.join(self.path_embed, "electra_config.json")
        if not path_vocab:
            path_vocab = os.path.join(self.path_embed, "vocab.txt")
        # 序列最大长度, 不得高于512
        kwargs["sequence_length"] = min(self.length_max, 512)
        # 加载bert等预训练模型
        self.model = build_transformer_model(return_keras_model=return_keras_model,
                                             max_position_embeddings=self.length_max,
                                             checkpoint_path=path_checkpoint,
                                             config_path=path_config,
                                             application=application,
                                             model=model,
                                             **kwargs
                                             )
        # 获取超参数config.json
        self.build_config(path_config)
        self.model_bert_config = self.model.get_config()
        # 获取bert4keras层输出
        num_hidden_layers = self.configs.get("num_hidden_layers", 12)
        self.output_layers = [self.model.get_layer('Transformer-{0}-FeedForward-Norm'.format(i)).output
                              for i in range(num_hidden_layers)]
        # merge layer of output, 输出层合并处理concat, add, pool-max or pool-avg
        self.build_merge_layer()
        # tokenizer
        self.build_tokenizer(path_vocab)


class Gpt2Embedding(BertEmbedding):
    def __init__(self, hyper_parameters):
        hyper_parameters["sharing"].update({"embed_size": 768,
                                            "token_type": "CHAR",
                                            "embed_type": "GPT2",
                                            })
        super().__init__(hyper_parameters)

    def build_transformer_model(self, return_keras_model: bool = True,
                                path_checkpoint: str = None,
                                path_config: str = None,
                                path_vocab: str = None,
                                application: str = "encoder",
                                model: str = "gpt2_ml",
                                **kwargs
                                ):
        # 地址
        if not path_checkpoint:
            path_checkpoint = os.path.join(self.path_embed, "gpt2_ml_model.ckpt")
        if not path_config:
            path_config = os.path.join(self.path_embed, "gpt2_ml_config.json")
        if not path_vocab:
            path_vocab = os.path.join(self.path_embed, "vocab.txt")
        # 序列最大长度, 不得高于512
        kwargs["sequence_length"] = min(self.length_max, 512)
        # 加载bert等预训练模型
        self.model = build_transformer_model(return_keras_model=return_keras_model,
                                             max_position_embeddings=self.length_max,
                                             checkpoint_path=path_checkpoint,
                                             config_path=path_config,
                                             application=application,
                                             model=model,
                                             **kwargs
                                             )
        # 获取超参数config.json
        self.build_config(path_config)
        self.config_model_bert = self.model.get_config()
        # 获取bert4keras层输出
        num_hidden_layers = self.configs.get("num_hidden_layers")
        output_layer = 'Transformer-FeedForward-Norm'
        self.output_layers = [self.model.get_layer(output_layer).get_output_at(i - 1)
                              for i in range(num_hidden_layers)]
        # merge layer of output, 输出层合并处理concat,add,pool-max or pool-avg
        self.build_merge_layer()
        # tokenizer
        self.build_tokenizer(path_vocab)


class RandomWordEmbedding(RandomEmbedding):
    def __init__(self, hyper_parameters):
        hyper_parameters["sharing"].update({"embed_size": 300,})
        super().__init__(hyper_parameters)

    def select_tokens_word2vec(self):
        """
        select tokens from token_dict or word2vec, 选择tokens
        Args:
            None
        Returns:
            None
        """
        self.word_vector = {}
        if self.path_embed:
            # 加载静态词向量
            self.word_vector = load_word2vec_format(path=self.path_embed, limit=self.limit)
            # 适配调整嵌入向量的维度
            self.embed_size = self.word_vector[list(self.word_vector.keys())[0]].size
        # 创建词向量矩阵, 存在则用预训练好的, 不存在则随机初始化, [PAD], [UNK], [BOS], [EOS]......
        embedding_matrix = []
        # 如果字典较小, 那么就不是只使用corpus语料的词语, 即word2vec里边的token全部使用
        if self.vocab_size >= 320:
            for k, v in self.token2idx.items():
                if k in self.word_vector:
                    v_word2vec = self.word_vector[k]
                else:
                    v_word2vec = np.random.uniform(-0.5, 0.5, self.embed_size)
                embedding_matrix.append(v_word2vec)
        # 只选择corpus里边的token, 可以防止embedding矩阵OOM
        else:
            embedding_matrix.append(np.zeros(self.embed_size))
            self.token2idx = self.token_dict.copy()
            for _ in range(len(self.token2idx)-1):
                embedding_matrix.append(np.random.uniform(-0.5, 0.5, self.embed_size))
            for k, v in self.word_vector.items():
                self.token2idx[k] = len(self.token2idx)
                embedding_matrix.append(self.word_vector[k])
        embedding_matrix = np.array(embedding_matrix)
        return embedding_matrix

    def select_tokens_random(self):
        """
        select tokens from token_dict or word2vec, 选择tokens
        Args:
            None
        Returns:
            None
        """
        # 创建词向量矩阵, 存在则用预训练好的, 不存在则随机初始化, [PAD], [UNK], [BOS], [EOS]......
        embedding_matrix = []
        # 第一个是[PAD], zeros全零
        embedding_matrix.append(np.zeros(self.embed_size))
        # 只选择corpus里边的token, 可以防止embedding矩阵OOM
        for _ in tqdm(range(len(self.token2idx)-1)):
            embedding_matrix.append(np.random.uniform(-0.5, 0.5, self.embed_size))
        embedding_matrix = np.array(embedding_matrix)
        return embedding_matrix

    def get_word_freq(self):
        """
        Get word freq and max-length from macropodus and embed, 从分词器和词嵌入class中获取已知词,最大成词长度
        Args:
            None
        Returns:
            None
        """
        from macropodus.segment import segs
        self.words_total = segs.dict_words_freq
        words_field = self.token2idx
        for wf in words_field.keys():
            if wf not in self.words_total:
                self.words_total[wf] = len(self.words_total)
        # self.words_total = words_common.update(words_field)
        self.len_word_max = 8
        for word in self.words_total:
            if is_total_chinese(word):
                len_word = len(word)
                if self.len_word_max < len_word:
                    self.len_word_max = len_word
        # 最大词语长度, 只计算中文

    def build_embedding(self, path_checkpoint: str = None, path_config: str = None, path_vocab: str = None,
                        path_embed: str = None, path_corpus: str = None, limit: int=None):
        """
        build embedding of word2vec or fasttext
        Args:
            path_checkpoint: No use
            path_config: No use
            path_vocab: No use
            path_embed: str, file path of word2vec, eg. "/home/embedding/word2vec/sgns.wiki.word"
            path_corpus: str, file path of train data, eg. "/home/corpus/ner_people_1998/train.json"
            limit: int, limit token of vocab, eg. 100
        Returns:
            None
        """
        # 训练语料地址(用于创建字/词典)
        if not path_corpus or not os.path.exists(path_corpus):
             path_corpus = self.train_data
        # 预训练embedding地址
        if path_embed and os.path.exists(path_embed):
            self.path_embed = path_embed
        # 限制加载词向量(词的个数)
        self.limit = limit
        # 构建词表字典
        self.build_tokenizer(path_corpus)
        # 字典大小
        self.vocab_size = len(self.token2idx)
        # 创建词向量矩阵, 存在则用预训练好的, 不存在则随机初始化, [PAD], [UNK], [BOS], [EOS]......
        if self.token_type.upper() == "WORD":
            embedding_matrix = self.select_tokens_word2vec()
            # 筛选后的词典大小/最大词语长度
            self.vocab_size = len(self.token2idx)
            self.get_word_freq()
            shape_input = (self.length_max, self.len_word_max, )
        else:
            embedding_matrix = self.select_tokens_random()
            # 筛选后的词典大小/最大词语长度
            self.vocab_size = len(self.token2idx)
            self.get_word_freq()
            shape_input = (self.length_max, )
        # 构建Tokenizer
        self.build_tokenizer_from_dict(self.token2idx)
        # embed
        self.input = L.Input(shape=shape_input, dtype="int32")
        self.output = L.Embedding(self.vocab_size,
                                  self.embed_size,
                                  input_length=self.length_max,
                                  weights=[embedding_matrix],
                                  trainable=self.trainable)(self.input)
        self.model = M.Model(self.input, self.output)


class MixEmbedding:
    def __init__(self, hyper_parameters: List[Dict]):
        self.hyper_parameters = hyper_parameters
        # self.embed_type_char = self.hyper_parameters_char.get("sharing", {}).get("embed_type", "RANDOM")
        # self.embed_type_word = self.hyper_parameters_word.get("sharing", {}).get("embed_type", "RANDOM")
        self.embed_type = "MIX"
        # self.token2idx = None

    def init_embed(self, hyper_parameters: Dict, path_corpus: str=None, path_embed: str=None) -> Any:
        embed_ed = RandomWordEmbedding(hyper_parameters)
        embed_ed.build_embedding(path_embed=path_embed, path_corpus=path_corpus)
        return embed_ed

    def exists_cover(self, hyper_parameters):
        path_embed_em = hyper_parameters.get("embed", {}).get("path_embed", None)
        train_data = hyper_parameters.get("data", {}).get("train_data", None)
        # 训练语料地址(用于创建字/词典)
        path_corpus = train_data
        # 预训练embedding地址
        path_embed = path_embed_em
        return path_corpus, path_embed

    def build_embedding(self, path_checkpoint: str = None, path_config: str = None, path_vocab: str = None,):
        self.hyper_parameters_char = self.hyper_parameters[0]
        self.hyper_parameters_word = self.hyper_parameters[1]
        # 加载 word-embed, char-embed
        path_corpus_char, path_embed_char = self.exists_cover(self.hyper_parameters_char)
        path_corpus_word, path_embed_word = self.exists_cover(self.hyper_parameters_word)
        self.embeded_char = self.init_embed(hyper_parameters=self.hyper_parameters_char,
                                            path_corpus=path_corpus_char,
                                            path_embed=path_embed_char, )
        self.embeded_word = self.init_embed(hyper_parameters=self.hyper_parameters_word,
                                            path_corpus=path_corpus_word,
                                            path_embed=path_embed_word, )
        self.input = [self.embeded_char.model.input, self.embeded_word.model.input]
        self.output = [self.embeded_char.model.output, self.embeded_word.model.output]
        self.model = M.Model(self.input, self.output)
        # self.model.summary(132)
        self.token2idx_char = self.embeded_char.token2idx
        self.token2idx_word = self.embeded_word.token2idx
        self.len_word_max = self.embeded_word.len_word_max
        self.words_total = self.embeded_word.words_total
        self.tokenizer_char = self.embeded_char.tokenizer
        self.tokenizer_word = self.embeded_word.tokenizer
        self.length_max = self.embeded_char.length_max
        self.task = self.embeded_char.task
        self.token2idx = {"token2idx_word": self.token2idx_word,
                          "token2idx_char": self.token2idx_char,
                          "len_word_max": self.len_word_max,
                          "words_total": self.words_total,
                          "length_max": self.length_max,
                          "task": self.task
                          }

    def build_tokenizer_from_dict(self, mix_dicts: Any):
        """
        reader and create tokenizer from dict, 构建字/词字典(从字典dict)
        Args:
            token2idx: dict, vocab , eg. {"token2idx_word": {}, "token2idx_char": {},
                                          "words_total": {}, "len_word_max": 32, "length_max": 32} 
        Returns:
            None
        """
        self.token2idx_word = mix_dicts.get("token2idx_word", {})
        self.token2idx_char = mix_dicts.get("token2idx_char", {})
        self.len_word_max = mix_dicts.get("len_word_max", 32)
        self.words_total = mix_dicts.get("words_total", {})
        self.length_max = mix_dicts.get("length_max", 32)
        self.task = mix_dicts.get("task", "SL")
        self.tokenizer_word = Tokenizer(self.token2idx_word, do_lower_case=True)
        self.tokenizer_char = Tokenizer(self.token2idx_char, do_lower_case=True)

    def cut_and_search(self, text: str):
        """
        cut search for Lattice-LSTM-Batch, 找到所有成词并返回
        Args:
            text: str, original inputs of text, eg. "macadam是什么！"   
        Returns:
            texts: List[List], eg. [[你好, 你是, 是谁], [你, 的, 名字]] 
        """
        len_text = len(text)
        texts = []
        # 双层遍历搜索(搜索引擎式成词)
        for i in range(len_text):
            word_piece = []
            # 第一个单词前面也要计算, [PAD], 保证与char级别的长度一致
            for j in range(i):
                word_select = text[j:i]
                if word_select in self.words_total:
                    word_piece.append(word_select)
            if not word_piece:
                word_piece = [PAD]
            # 最大个词的个数不会大于成词最大长度, 因为数据是以最后一个字结尾的
            word_piece = word_piece if self.len_word_max<=len(word_piece) \
                                    else word_piece + [PAD] * (self.len_word_max-len(word_piece))

            texts.append(word_piece)
        return texts

    def get_wc_lstm_words(self, text: str = None, second_text: Any = None,):
        """
        tramsformer text to LATTICE-LSTM-BATCH-words, 将文本转化为LATTICE-LSTM-BATCH模型需要的格式
        Args:
            text: str, first input of sentence when in single-task, pair-task or multi-task, eg. "macadam英文什么意思"
            second_text: List, second inputs of sentence, eg. ["macadam?", "啥macadam?", "macadam什么意思"] 
        Returns:
            List[List]
        """
        second_text_wc = [self.cut_and_search(st) for st in second_text] if second_text else second_text
        text_wc = self.cut_and_search(text)
        return text_wc, second_text_wc

    def sent2idx_mix(self, text: Any = None, second_text: Any = None, limit_lengths: List = None,
                           use_seconds: bool = True, is_multi: bool = True):
        """
        cut/index/padding sentence, 切词/数字转化/pad 文本
        Args:
            text: Any, first input of sentence when in single-task, pair-task or multi-task, eg. "macadam英文什么意思"
            second_text: Any, second inputs of sentence, eg. ["macadam?", "啥macadam?", "macadam什么意思"] 
            limit_lengths: List, limit lengths of each in texts(second inputs), eg.[36, 36,128]
            use_seconds: bool, whether use class"encode_multi" or not
            is_multi: bool, whether sign sentence in texts with multi or not
        Returns:
            List
        """
        input_id, id_type = self.tokenizer_word.encode_mix(first_text=text, second_texts=second_text,
                                                           limit_lengths=limit_lengths, length_max=self.length_max,
                                                           is_multi=is_multi)

        input_id = input_id + [[self.tokenizer_word.token_to_id(WC)]*self.len_word_max] * (self.length_max - len(input_id))
        id_type = id_type.extend([0] * (self.length_max - len(input_id)))
        return [input_id, id_type]

    def sent2idx_char(self, text: Any = None, second_text: Any = None, limit_lengths: List = None,
                      use_seconds: bool = True, is_multi: bool = True):
        """
        cut/index/padding sentence, 切词/数字转化/pad 文本
        Args:
            text: Any, first input of sentence when in single-task, pair-task or multi-task, eg. "macadam英文什么意思"
            second_text: Any, second inputs of sentence, eg. ["macadam?", "啥macadam?", "macadam什么意思"] 
            limit_lengths: List, limit lengths of each in texts(second inputs), eg.[36, 36,128]
            use_seconds: bool, whether use class"encode_multi" or not
            is_multi: bool, whether sign sentence in texts with multi or not
        Returns:
            List
        """
        text = list(text)
        second_text = [list(st) for st in second_text] if second_text else None
        if use_seconds: # 第二个句子传入多个, 即定义多个[SEP]
            input_id, id_type = self.tokenizer_char.encode_average(first_text=text,
                                                                   second_texts=second_text,
                                                                   limit_lengths=limit_lengths,
                                                                   length_max=self.length_max,
                                                                   is_multi=is_multi)
            # input_mask = [0 if ids == 0 else 1 for ids in input_id]
            idxs = [input_id, id_type]
        else:
            input_id, input_type_id = self.tokenizer_char.encode(first_text=text,
                                                            second_text=second_text,
                                                            max_length=self.length_max)
            # input_mask = [0 if ids == 0 else 1 for ids in input_id]
            idxs = [input_id, input_type_id]
        idxs = padding_sequences(sequences=[idxs] if type(idxs[0]) == int else idxs, task=self.task,
                                 padding_start=self.token2idx_char[CLS], length_max=self.length_max,
                                 padding_end=self.token2idx_char[SEP], padding=self.token2idx_char[PAD],)
        return idxs

    def sent2idx(self, text: Any = None, second_text: Any = None, limit_lengths: List = None,
                 use_seconds: bool = True, is_multi: bool = True):
        """
        encode sentence, 文本编码(model.predict)
        Args:
            text: str, first input of sentence when in single-task, pair-task or multi-task, eg. "macadam英文什么意思"
            second_text: List, second inputs of sentence, eg. ["macadam?", "啥macadam?", "macadam什么意思"] 
            limit_lengths: List, limit lengths of each in texts(second inputs), eg.[36, 36,128]
            use_seconds: bool, whether use class"encode_multi" or not
            is_multi: bool, whether sign sentence in texts with multi or not
        Returns:
            List[List]
        """
        idxs_char = self.sent2idx_char(text, second_text, limit_lengths, use_seconds, is_multi)
        text, second_text = self.get_wc_lstm_words(text, second_text)
        idxs_word = self.sent2idx_mix(text, second_text, limit_lengths, use_seconds, is_multi)
        return [idxs_char, idxs_word]

    def encode(self, text: Any = None, second_text: Any = None, limit_lengths: List = None,
               use_seconds: bool = True, is_multi: bool = True) -> List[List]:
        """
        cut/index/padding sentence, 切词/数字转化/pad 文本
        Args:
            text: Any, first input of sentence when in single-task, pair-task or multi-task, eg. "macadam英文什么意思"
            second_text: List, second inputs of sentence, eg. ["macadam?", "啥macadam?", "macadam什么意思"] 
            limit_lengths: List, limit lengths of each in texts(second inputs), eg.[36, 36,128]
            use_seconds: bool, whether use class"encode_multi" or not
            is_multi: bool, whether sign sentence in texts with multi or not
        Returns:
            List
        """
        sent_ids = self.sent2idx(text=text, second_text=second_text, is_multi=is_multi,
                                 limit_lengths=limit_lengths, use_seconds=use_seconds,)
        sent_idx_np_char = np.array([sent_ids[0][0], sent_ids[0][0]])
        sent_idx_np_word = np.array([sent_ids[1][0], sent_ids[1][0]])
        # res_char = self.embeded_char.model.predict(sent_idx_np_char)
        # res_word = self.embeded_word.model.predict(sent_idx_np_word)
        res_all = self.model.predict([sent_idx_np_char, sent_idx_np_word])
        # return [res_char, res_word, res_all]
        return res_all


embedding_map = {"ROBERTA": RoBertaEmbedding,
                 "ELECTRA": ElectraEmbedding,
                 "RANDOM": RandomEmbedding,
                 "ALBERT": AlBertEmbedding,
                 "XLNET": XlnetEmbedding,
                 "NEZHA": NezhaEmbedding,
                 "GPT2": Gpt2Embedding,
                 "WORD": WordEmbedding,
                 "BERT": BertEmbedding,
                 "MIX": MixEmbedding
                 }


if __name__ == '__main__':
    ###### tet bert-embedding
    os.environ["TF_KERAS"] = "1"
    # bert预训练模型路径
    path_embed = "D:/soft_install/dataset/bert-model/chinese_L-12_H-768_A-12"
    path_check_point = path_embed + "/bert_model.ckpt"
    path_config = path_embed + "/bert_config.json"
    path_vocab = path_embed + "/vocab.txt"
    params = {"embed": {"path_embed": path_embed,
                        "layer_idx": [0, -2, -1],
                        "merge_type": "pool-max", # "concat", "add", "pool-max", "pool-avg", "multi"
                        },
              "sharing": {"length_max": 32},
              }
    bert_embed = BertEmbedding(params)
    # 内部定义了默认的ckpt/config/vocab文件名, 如果不对则先定义os.environ["MACADAM_LEVEL"] = "CUSTOM"模式, 用bert_embed.build_embedding
    bert_embed.build_embedding(path_checkpoint=path_check_point,
                               path_config=path_config,
                               path_vocab=path_vocab)
    res = bert_embed.encode(text="macadam怎么翻译", second_text="macadam是碎石路")
    print(res)
    while True:
        print("请输入first_text:")
        first_text = input()
        print("请输入second_text:")
        second_text = input()
        res = bert_embed.encode(text=first_text, second_text=second_text)
        print(res)

    mm = 0
