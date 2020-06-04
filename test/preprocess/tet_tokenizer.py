# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/4/30 21:49
# @author  : Mo
# @function: test preprocess of macadam


from macadam.base.preprocess import Tokenizer4Macadam
from typing import List, Any



def truncate_sequence_multi(length_max: int, sequences: List[Any], pop_index: int = -2):
    """
    truncate sequence of multi, 均衡裁剪List[List]数据
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


def _tet_tokenizer4macadm():
    """测试tokenizer, 即bert等输入多个句子的情况"""
    path_vocab = "D:/soft_install/dataset/bert-model/chinese_L-12_H-768_A-12/vocab.txt"
    import codecs
    # 读取字典文件
    token2idx = {}
    with codecs.open(path_vocab, "r", "utf-8") as reader:
        for line in reader:
            token = line.strip()
            token2idx[token] = len(token2idx)
    # tokenizer4初始化
    tokenizer4 = Tokenizer4Macadam(token2idx)
    # tokenizer = Tokenizer(vocab_dict)

    sent1 = "你是谁1" * 2
    sent2 = "我是谁2" * 2
    sent3 = "他是谁3" * 300

    tok1 = tokenizer4.encode(first_text=sent1, second_text=sent2 + sent3, max_length=32)
    tok2 = tokenizer4.encode_multi(first_text=sent1, second_texts=[sent2, sent3], length_max=32, is_multi=True)
    # 测试tet, string
    tok3 = tokenizer4.encode_average(first_text=sent1, second_texts=[sent2, sent3], length_max=32, is_multi=True)
    tok31 = tokenizer4.encode_average(first_text=sent1, second_texts=[], length_max=523, is_multi=True)
    tok32 = tokenizer4.encode_average(first_text=sent1, second_texts=[sent2], length_max=523, is_multi=True)
    tok33 = tokenizer4.encode_average(first_text=sent1, second_texts=[sent1, sent2, sent3], length_max=523, is_multi=True)
    tok34 = tokenizer4.encode_average(first_text=sent1, second_texts=None, length_max=523, is_multi=True)
    # 测试tet, list
    sent1 = list(sent1)
    sent2 = list(sent2)
    sent3 = list(sent3)
    tok30 = tokenizer4.encode_average(first_text=sent1, second_texts=[sent2, sent3], length_max=32, is_multi=True)
    tok310 = tokenizer4.encode_average(first_text=sent1, second_texts=[], length_max=523, is_multi=True)
    tok320 = tokenizer4.encode_average(first_text=sent1, second_texts=[sent2], length_max=523, is_multi=True)
    tok330 = tokenizer4.encode_average(first_text=sent1, second_texts=[sent1, sent2, sent3], length_max=523, is_multi=True)
    tok340 = tokenizer4.encode_average(first_text=sent1, second_texts=None, length_max=523, is_multi=True)
    print(tok1)
    print(tok2)
    print(tok3)
    print(tok31)
    print(tok32)
    print(tok33)
    print(tok34)


if __name__ == '__main__':

    seqs = [["尼亚", "BBC"]*129, ["效力"]*160, ["BDP"]*10000]
    truncate_sequence_multi(128, seqs)

    mm = 0

    _tet_tokenizer4macadm()

    mm = 0
