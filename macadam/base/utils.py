# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/4/23 10:17
# @author  : Mo
# @function: tools of util of Macadam


from macadam.conf.logger_config import logger
from collections import Counter, OrderedDict
from typing import Union, Dict, List, Any
import numpy as np
import macropodus
import json
import os


__all_ = ["load_word2vec_format",
          "padding_sequences",
          "is_total_chinese",
          "metrics_report",
          "macropodus_cut",
          "delete_file",
          "get_ngram",
          "save_json",
          "load_json",
          "txt_write",
          "txt_read",
          "dict_sort"
          ]


def load_word2vec_format(path: str,
                         encoding: str = "utf-8",
                         dtype: Union[np.float32, np.float16] = np.float16,
                         limit: int = None) -> Dict:
    """
    Load word2vec from word2vec-format file, 加载词向量文件
    Args:
        path: file path of saved word2vec-format file.
        encoding: If you save the word2vec model using non-utf8 encoding for words, specify that encoding in `encoding`.
        dtype: Can coerce dimensions to a non-default float type (such as `np.float16`) to save memory.
        limit: the limit of the words of word2vec
    Returns:
        dict of word2vec
    """
    w2v = {}
    count = 0
    with open(path, "r", encoding=encoding) as fr:
        for line in fr:
            count += 1
            if count > 1:  # 第一条不取
                idx = line.index(" ")
                word = line[:idx]
                vecotr = line[idx + 1:]
                vector = np.fromstring(vecotr, sep=" ", dtype=dtype)
                w2v[word] = vector
            # limit限定返回词向量个数
            if limit and count >= limit:
                break
    return w2v


def delete_file(path_dir: str):
    """
    Delete model files in the directory, eg. h5/json/pb 
    Args:
        path_dir: path of directory, where many files in the directory
    """
    for i in os.listdir(path_dir):
        # 取文件或者目录的绝对路径
        path_children = os.path.join(path_dir, i)
        if os.path.isfile(path_children):
            if path_children.endswith(".h5") or path_children.endswith(".json") or path_children.endswith(".pb"):
                os.remove(path_children)
        else:  # 递归, 删除目录下的所有文件
            delete_file(path_children)


def txt_read(path: str, encoding: str = "utf-8") -> List[str]:
    """
    Read Line of list form file
    Args:
        path: path of save file, such as "txt"
        encoding: type of encoding, such as "utf-8", "gbk"
    Returns:
        dict of word2vec, eg. {"macadam":[...]}
    """

    lines = []
    try:
        file = open(path, "r", encoding=encoding)
        lines = file.readlines()
        file.close()
    except Exception as e:
        logger.info(str(e))
    finally:
        return lines


def txt_write(lines: List[str],
              path: str,
              model: str = "w",
              encoding: str = "utf-8"):
    """
    Write Line of list to file
    Args:
        lines: lines of list<str> which need save
        path: path of save file, such as "txt"
        model: type of write, such as "w", "a+"
        encoding: type of encoding, such as "utf-8", "gbk"
    """

    try:
        file = open(path, model, encoding=encoding)
        file.writelines(lines)
        file.close()
    except Exception as e:
        logger.info(str(e))


def save_json(lines: Union[List, Dict],
              path: str,
              encoding: str = "utf-8",
              indent: int = 4):
    """
    Write Line of List<json> to file
    Args:
        lines: lines of list[str] which need save
        path: path of save file, such as "json.txt"
        encoding: type of encoding, such as "utf-8", "gbk"
    """
    
    with open(path, "w", encoding=encoding) as fj:
        fj.write(json.dumps(lines, ensure_ascii=False, indent=indent))
    fj.close()


def load_json(path: str, encoding: str="utf-8") -> Union[List, Any]:
    """
    Read Line of List<json> form file
    Args:
        path: path of save file, such as "txt"
        encoding: type of encoding, such as "utf-8", "gbk"
    Returns:
        model_json: dict of word2vec, eg. [{"大漠帝国":132}]
    """
    with open(path, "r", encoding=encoding) as fj:
        model_json = json.load(fj)
        fj.close()
    return model_json


def macropodus_cut(text: str) -> List:
    """
    cut words of chinese using macropodus
    Args:
        text: text of string which need cws, eg. "大漠帝国是谁"
    Returns:
        list of words, eg. ["大漠帝国", "是", "谁"]
    """
    return list(macropodus.cut(text))


def get_ngram(text: str, ns: List[int] = [1]) -> List[List]:
    """
    get n-gram from text
    Args:
        text: text of string which need cws, eg. "是谁"
    Returns:
        list of words, eg. ["是", "谁"]
    """
    if type(ns) != list:
        raise RuntimeError("ns of function get_ngram() must be list!")
    for n in ns:
        if n < 1:
            raise RuntimeError("enum of ns must '>1'!")
    len_text = len(text)
    ngrams = []
    for n in ns: # 遍历, eg. n=1-5
        ngram_n = []
        for i in range(len_text):
            if i + n <= len_text:
                ngram_n.append(text[i:i+n])
            else:
                break
        if not ngram_n:
            ngram_n.append(text)
        ngrams += ngram_n
    return ngrams


def padding_sequences(sequences: List, length_max: int=32, padding: Any=0, task: str="TC",
                      padding_start: str="O", padding_end: str="O") -> List[List]:
    """
    padding sequence to the same length of max, 截断与补长
    Args:
        sequences: List of sequence, eg. [[1, 3, 5], [2, 4, 6, 8]]
        length_max: length of max of which we hope padding
        padding: padding symbol, eg. 1
    Returns:
        list of sequence, eg. [[1, 3, 5, 0], [2, 4, 6, 8]]
    """
    # if task == "TC":
    #     pad_sequences = [seq[:length_max] if len(seq)>=length_max
    #                      else seq+[padding] * (length_max - len(seq))
    #                      for seq in sequences]
    # else: # 判断条件为length_max-2(start, end)
    #     pad_sequences = [[padding_start] + seq[:length_max-2] + [padding_end] if len(seq) >= length_max-2
    #                          else [padding_start] + seq + [padding] * (length_max - len(seq) - 2) + [padding_end]
    #                          for seq in sequences]
    pad_sequences = [seq[:length_max] if len(seq) >= length_max
                     else seq + [padding] * (length_max - len(seq))
                     for seq in sequences]
    return pad_sequences


def metrics_report(y_true: List, y_pred: List, rounded: int=6, epsilon:float=1e-9, use_draw: bool=True):
    """
    calculate metrics and draw picture, 计算评估指标并画图
    code: some code from: 
    Args:
        y_true: list, label of really true, eg. ["TRUE", "FALSE"]
        y_pred: list, label of model predict, eg. ["TRUE", "FALSE"]
        rounded: int, bit you reserved , eg. 6
        epsilon: float, ε, constant of minimum, eg. 1e-6
        use_draw: bool, whether draw picture or not, True
    Returns:
        metrics(precision, recall, f1, accuracy, support), report
    """

    def calculate_metrics(datas: Dict) -> Any:
        """
        calculate metrics after of y-true nad y-pred, 计算准确率, 精确率, 召回率, F1-score
        Args:
            datas: Dict, eg. {"TP": 5, "FP": 3, "TN": 8, "FN": 9}
        Returns:
            accuracy, precision, recall, f1
        """
        # accuracy = datas["TP"] / (datas["label_counter"] + epsilon)
        # accuracy = (datas["TP"] + datas["TN"]) / (datas["TP"] + datas["TN"] + datas["FP"] + datas["FN"] + epsilon)
        precision = datas["TP"] / (datas["TP"] + datas["FP"] + epsilon)
        recall = datas["TP"] / (datas["TP"] + datas["FN"] + epsilon)
        f1 = (precision * recall * 2) / (precision + recall + epsilon)
        # accuracy = round(accuracy, rounded)
        precision = round(precision, rounded)
        recall = round(recall, rounded)
        f1 = round(f1, rounded)
        # return accuracy, precision, recall, f1
        return precision, recall, f1

    label_counter = dict(Counter(y_true))
    # 统计每个类下的TP, FP, TN, FN等信息, freq of some (one label)
    freq_so = {yti: {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for yti in sorted(set(y_true))}
    for idx in range(len(y_true)):
        correct = y_true[idx]
        predict = y_pred[idx]
        # true
        if correct == predict:
            for k, v in freq_so.items():
                if k == correct:
                    freq_so[correct]["TP"] += 1
                else:
                    freq_so[k]["TN"] += 1
        # flase
        else:
            for k, v in freq_so.items():
                if k == correct:
                    freq_so[correct]["FN"] += 1
                elif k == predict:
                    freq_so[k]["FP"] += 1
                else:
                    freq_so[k]["TN"] += 1
    # 统计每个类下的评估指标("accuracy", "precision", "recall", "f1", "support")
    metrics = {}
    freq_to = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    keys = list(freq_to.keys())
    for k, v in freq_so.items():
        for key in keys:
            freq_to[key] += v[key]
        # v["label_counter"] = label_counter[k]
        precision, recall, f1 = calculate_metrics(v)
        metrics[k] = {"precision": precision,
                      "recall": recall,
                      "f1-score": f1,
                      # "accuracy": accuracy,
                      "support": label_counter[k]}
    # 计算平均(mean)评估指标
    mean_metrics = {}
    for mmk in list(metrics.values())[0].keys():
        for k, _ in metrics.items():
            k_score = sum([v[mmk] for k, v in metrics.items()]) / len(metrics)
            mean_metrics["mean_{}".format(mmk)] = round(k_score, rounded)
    # 计算总计(sum)评估指标
    # freq_to["label_counter"] = sum(label_counter.values())
    sum_precision, sum_recall, micro_f1 = calculate_metrics(freq_to)
    metrics['mean'] = mean_metrics
    metrics['sum'] = {"sum_precision": sum_precision,
                      "sum_recall": sum_recall,
                      "sum_f1-score": micro_f1,
                      # "sum_accuracy": sum_accuracy,
                      "sum_support": sum(label_counter.values())
                      }
    report = None
    if use_draw:
        # 打印头部
        sign_tol = ["mean", "sum"]
        labels = list(label_counter.keys())
        target_names = [u"%s" % l for l in labels] + sign_tol
        name_width = max(len(cn) for cn in target_names)
        width = max(name_width, rounded)
        headers = ["precision", "recall", "f1-score", "support"]
        head_fmt = u"{:>{width}} " + u" {:>9}" * len(headers)
        report = "\n\n" + head_fmt.format("", *headers, width=width)
        report += "\n\n"
        # 具体label的评估指标
        row_fmt = u"{:>{width}} " + u" {:>9.{rounded}f}" * (len(headers)-1) + u" {:>9}\n"
        for li in labels:
            [p, r, f1, s] = [metrics[li][hd] for hd in headers]
            report += row_fmt.format(li, p, r, f1, int(s), width=width, rounded=rounded)
        report += "\n"
        # 评估指标sum, mean
        for sm in sign_tol:
            [p, r, f1, s] = [metrics[sm][sm + "_" + hd] for hd in headers]
            report += row_fmt.format(sm, p, r, f1, int(s), width=width, rounded=rounded)
        report += "\n"
        # logger.info(report)

    return metrics, report


def is_total_chinese(text: str) -> bool:
    """
    judge is total chinese or not, 判断是不是全是中文
    Args:
        text: str, eg. "macadam, 碎石路"
    Returns:
        bool, True or False
    """
    for word in text:
        if not "\u4e00" <= word <= "\u9fa5":
            return False
    return True


def dict_sort(in_dict: dict)-> OrderedDict:
    """
    sort dict by values, 给字典排序(依据值大小)
    Args:
        in_dict: dict, eg. {"游戏:132, "旅游":32}
    Returns:
        OrderedDict
    """
    in_dict_sort = sorted(in_dict.items(), key=lambda x:x[1], reverse=True)
    return OrderedDict(in_dict_sort)


if __name__ == "__main__":
    # # load_word2vec_format
    # word2vec = load_word2vec_format("w2v_model_merge_short.vec")
    # mm = 0

    # # padding_sequences
    # seq = [[1]*32, [1]*569, [1]*512]
    # seq_pad = padding_sequences(seq, length_max=512, padding=0)

    # metrics report
    # y_true = ["yes", "no", "pos", "yes", "no", "pos", "yes", "no", "pos", "yes", "no", "pos", "pos"]
    # y_pred = ["yes", "yes", "pos", "no", "no", "no", "yes", "no", "pos", "yes", "no", "pos", "pos"]
    y_true = [1, 2, 3, 4, 3, 5]
    y_pred = [1, 2, 2, 2, 3, 5]
    res, report = metrics_report(y_true, y_pred)
    print(res)
    print(report)
    # from sklearn.metrics import classification_report
    # res2 = classification_report(y_true, y_pred)
    # print(res2)

    mm = 0
