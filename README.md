<p align="center">
    <img src="test/images/macadam_logo.png" width="480"\>
</p>

# [Macadam](https://github.com/yongzhuo/Macadam)

[![PyPI](https://img.shields.io/pypi/v/Macadam)](https://pypi.org/project/Macadam/)
[![Build Status](https://travis-ci.com/yongzhuo/Macadam.svg?branch=master)](https://travis-ci.com/yongzhuo/Macadam)
[![PyPI_downloads](https://img.shields.io/pypi/dm/Macadam)](https://pypi.org/project/Macadam/)
[![Stars](https://img.shields.io/github/stars/yongzhuo/Macadam?style=social)](https://github.com/yongzhuo/Macadam/stargazers)
[![Forks](https://img.shields.io/github/forks/yongzhuo/Macadam.svg?style=social)](https://github.com/yongzhuo/Macadam/network/members)
[![Join the chat at https://gitter.im/yongzhuo/Macadam](https://badges.gitter.im/yongzhuo/Macadam.svg)](https://gitter.im/yongzhuo/Macadam?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
>>> Macadam是一个以Tensorflow(Keras)和bert4keras为基础，专注于文本分类、序列标注和关系抽取的自然语言处理工具包。支持RANDOM、WORD2VEC、FASTTEXT、BERT、ALBERT、ROBERTA、NEZHA、XLNET、ELECTRA、GPT-2等EMBEDDING嵌入;
    支持FineTune、FastText、TextCNN、CharCNN、BiRNN、RCNN、DCNN、CRNN、DeepMoji、SelfAttention、HAN、Capsule等文本分类算法; 
    支持CRF、Bi-LSTM-CRF、CNN-LSTM、DGCNN、Bi-LSTM-LAN、Lattice-LSTM-Batch、MRC等序列标注算法。


## 目录
* [安装](#安装)
* [数据](#数据)
* [使用方式](#使用方式)
* [TODO](#TODO)
* [paper](#paper)
* [参考](#参考)


# 安装 
```bash
pip install Macadam

# 清华镜像源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple Macadam
```


# 数据
## 数据来源
  * [ner_clue_2020](https://github.com/CLUEbenchmark/CLUENER2020), CLUENER2020中文细粒度命名实体识别
  * [ner_people_1998](http://www.icl.pku.edu.cn/icl_res/), 《人民日报》标注语料库中的语料, 1998.01
  * [baidu_qa_2019](https://github.com/liuhuanyong/MiningZhiDaoQACorpus), 百度知道问答语料
  * [thucnews](http://thuctc.thunlp.org/), 新浪新闻RSS订阅频道2005-2011年间的历史数据筛
## 数据格式
```
1. 文本分类  (txt格式, 每行为一个json):

{"x": {"text": "人站在地球上为什么没有头朝下的感觉", "texts2": []}, "y": ["教育"]}
{"x": {"text": "我的小baby", "texts2": []}, "y": ["娱乐"]}
{"x": {"text": "请问这起交通事故是谁的责任居多小车和摩托车发生事故在无红绿灯", "texts2": []}, "y": ["娱乐"]}

2. 序列标注 (txt格式, 每行为一个json):

{"x": {"text": "海钓比赛地点在厦门与金门之间的海域。", "texts2": []}, "y": ["O", "O", "O", "O", "O", "O", "O", "B-LOC", "I-LOC", "O", "B-LOC", "I-LOC", "O", "O", "O", "O", "O", "O"]}
{"x": {"text": "参加步行的有男有女，有年轻人，也有中年人。", "texts2": []}, "y": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]}
{"x": {"text": "山是稳重的，这是我最初的观念。", "texts2": []}, "y": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]}
{"x": {"text": "立案不结、以罚代刑等问题有较大改观。", "texts2": []}, "y": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]}

```

# 使用方式
  更多样例sample详情见test目录
## 文本分类, text-classification
```bash
# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/5/8 21:33
# @author  : Mo
# @function: test trainer of bert


# 适配linux
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(path_root)
# cpu-gpu与tf.keras
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_KERAS"] = "1"
# macadam
from macadam.conf.path_config import path_root, path_tc_baidu_qa_2019, path_tc_thucnews
from macadam.tc import trainer


if __name__=="__main__":
    # bert-embedding地址, 必传
    path_embed = "D:/soft_install/dataset/bert-model/chinese_L-12_H-768_A-12"
    path_checkpoint = path_embed + "/bert_model.ckpt"
    path_config = path_embed + "/bert_config.json"
    path_vocab = path_embed + "/vocab.txt"

    # 训练/验证数据地址, 必传
    # path_train = os.path.join(path_tc_thucnews, "train.json")
    # path_dev = os.path.join(path_tc_thucnews, "dev.json")
    path_train = os.path.join(path_tc_baidu_qa_2019, "train.json")
    path_dev = os.path.join(path_tc_baidu_qa_2019, "dev.json")

    # 网络结构, 嵌入模型, 大小写都可以, 必传
    # 网络模型架构(Graph), "FineTune", "FastText", "TextCNN", "CharCNN",
    # "BiRNN", "RCNN", "DCNN", "CRNN", "DeepMoji", "SelfAttention", "HAN", "Capsule"
    network_type = "TextCNN"
    # 嵌入(embedding)类型, "ROBERTA", "ELECTRA", "RANDOM", "ALBERT", "XLNET", "NEZHA", "GPT2", "WORD", "BERT"
    embed_type = "BERT"
    # token级别, 一般为"char", 只有random和word的embedding时存在"word"
    token_type = "CHAR"
    # 任务, "TC"(文本分类), "SL"(序列标注), "RE"(关系抽取)
    task = "TC"
    
    # 模型保存目录, 必传
    path_model_dir = os.path.join(path_root, "data", "model", network_type)
    # 开始训练, 可能前几轮loss较大acc较低, 后边会好起来
    trainer(path_model_dir, path_embed, path_train, path_dev, path_checkpoint, path_config, path_vocab,
            network_type=network_type, embed_type=embed_type, token_type=token_type, task=task)
    mm = 0
```
## 序列标注, sequence-labeling
```bash
# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/5/8 21:33
# @author  : Mo
# @function: test trainer of bert


# 适配linux
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(path_root)
## cpu-gpu与tf.keras
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_KERAS"] = "1"

# 地址, tf.keras
from macadam.conf.path_config import path_embed_bert, path_embed_word2vec_word, path_embed_word2vec_char
from macadam.conf.path_config import path_root, path_ner_people_1998, path_ner_clue_2020
from macadam.sl import trainer


if __name__=="__main__":
    # bert-embedding地址, 必传
    path_embed = path_embed_bert # path_embed_bert, path_embed_word2vec_word, path_embed_word2vec_char
    path_checkpoint = os.path.join(path_embed + "bert_model.ckpt")
    path_config = os.path.join(path_embed + "bert_config.json")
    path_vocab = os.path.join(path_embed + "vocab.txt")

    # # 训练/验证数据地址
    # path_train = os.path.join(path_ner_people_1998, "train.json")
    # path_dev = os.path.join(path_ner_people_1998, "dev.json")
    path_train = os.path.join(path_ner_clue_2020, "ner_clue_2020.train")
    path_dev = os.path.join(path_ner_clue_2020, "ner_clue_2020.dev")
    # 网络结构
    # "CRF", "Bi-LSTM-CRF", "Bi-LSTM-LAN", "CNN-LSTM", "DGCNN", "LATTICE-LSTM-BATCH"
    network_type = "CRF"
    # 嵌入(embedding)类型, "ROOBERTA", "ELECTRA", "RANDOM", "ALBERT", "XLNET", "NEZHA", "GPT2", "WORD", "BERT"
    # MIX,  WC_LSTM时候填两个["RANDOM", "WORD"], ["WORD", "RANDOM"], ["RANDOM", "RANDOM"], ["WORD", "WORD"]
    embed_type = "RANDOM" 
    token_type = "CHAR"
    task = "SL"
    lr = 1e-5 if embed_type in ["ROBERTA", "ELECTRA", "ALBERT", "XLNET", "NEZHA", "GPT2", "BERT"] else 1e-3
    # 模型保存目录, 如果不存在则创建
    path_model_dir = os.path.join(path_root, "data", "model", network_type)
    if not os.path.exists(path_model_dir):
        os.mkdir(path_model_dir)
    # 开始训练
    trainer(path_model_dir, path_embed, path_train, path_dev,
            path_checkpoint, path_config, path_vocab,
            network_type=network_type, embed_type=embed_type,
            task=task, token_type=token_type,
            is_length_max=False, use_onehot=False, use_file=False, use_crf=True,
            layer_idx=[-2], learning_rate=lr,
            batch_size=30, epochs=12, early_stop=6, rate=1)
    mm = 0
```
  
# TODO
 * 文本分类TC(TextGCN)
 * 序列标注SL(MRC)
 * 关系抽取RE
 * 嵌入embed(xlnet)


# paper
## 文本分类(TC, text-classification)
* FastText:   [Bag of Tricks for Efﬁcient Text Classiﬁcation](https://arxiv.org/abs/1607.01759)
* TextCNN：   [Convolutional Neural Networks for Sentence Classiﬁcation](https://arxiv.org/abs/1408.5882)
* charCNN-kim：   [Character-Aware Neural Language Models](https://arxiv.org/abs/1508.06615)
* charCNN-zhang:  [Character-level Convolutional Networks for Text Classiﬁcation](https://arxiv.org/pdf/1509.01626.pdf)
* TextRNN：   [Recurrent Neural Network for Text Classification with Multi-Task Learning](https://www.ijcai.org/Proceedings/16/Papers/408.pdf)
* RCNN：      [Recurrent Convolutional Neural Networks for Text Classification](http://www.nlpr.ia.ac.cn/cip/~liukang/liukangPageFile/Recurrent%20Convolutional%20Neural%20Networks%20for%20Text%20Classification.pdf)
* DCNN:       [A Convolutional Neural Network for Modelling Sentences](https://arxiv.org/abs/1404.2188)
* DPCNN:      [Deep Pyramid Convolutional Neural Networks for Text Categorization](https://www.aclweb.org/anthology/P17-1052)
* VDCNN:      [Very Deep Convolutional Networks](https://www.aclweb.org/anthology/E17-1104)
* CRNN:        [A C-LSTM Neural Network for Text Classification](https://arxiv.org/abs/1511.08630)
* DeepMoji:    [Using millions of emojio ccurrences to learn any-domain represent ations for detecting sentiment, emotion and sarcasm](https://arxiv.org/abs/1708.00524)
* SelfAttention: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* HAN: [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)
* CapsuleNet: [Dynamic Routing Between Capsules](https://arxiv.org/pdf/1710.09829.pdf)
* Transformer(encode or decode): [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* Bert:                  [BERT: Pre-trainingofDeepBidirectionalTransformersfor LanguageUnderstanding]()
* Xlnet:                 [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237)
* Albert:                [ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS](https://arxiv.org/pdf/1909.11942.pdf)
* RoBERTa:               [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
* ELECTRA:               [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://openreview.net/pdf?id=r1xMH1BtvB)
* TextGCN:               [Graph Convolutional Networks for Text Classification](https://arxiv.org/abs/1809.05679)


## 序列标注(SL, sequence-labeling)
* CRF:            [Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers)
* Bi-LSTM-CRF:    [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/pdf/1508.01991.pdf)
* CNN-LSTM:       [End-to-endSequenceLabelingviaBi-directionalLSTM-CNNs-CRF](https://arxiv.org/abs/1603.01354)
* DGCNN:          [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122)
* Bi-LSTM-LAN:    [Hierarchically-Reﬁned Label Attention Network for Sequence Labeling](https://arxiv.org/abs/1908.08676v2)
* LATTICE-LSTM-BATCH:    [An Encoding Strategy Based Word-Character LSTM for Chinese NER](https://www.aclweb.org/anthology/N19-1247/)
* MRC:            [A Unified MRC Framework for Named Entity Recognition](https://arxiv.org/abs/1910.11476v2)


# 参考
* keras与tensorflow版本对应: [https://docs.floydhub.com/guides/environments/](https://docs.floydhub.com/guides/environments/)
* bert4keras:   [https://github.com/bojone/bert4keras](https://github.com/bojone/bert4keras)
* Kashgari: [https://github.com/BrikerMan/Kashgari](https://github.com/BrikerMan/Kashgari)
* fastNLP: [https://github.com/fastnlp/fastNLP](https://github.com/fastnlp/fastNLP)
* HanLP: [https://github.com/hankcs/HanLP](https://github.com/hankcs/HanLP)


# Reference
For citing this work, you can refer to the present GitHub project. For example, with BibTeX:
```
@misc{Macadam,
    howpublished = {url{https://github.com/yongzhuo/Macadam}},
    title = {Macadam},
    author = {Yongzhuo Mo},
    publisher = {GitHub},
    year = {2020}
}
```
*希望对你有所帮助!

