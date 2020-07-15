# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/5/6 21:22
# @author  : Mo
# @function: original corpus of text-classification change to standardized, 将训练数据语料格式转为macadam需要的格式


# 适配linux
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(path_root)
# macadam
from macadam.base.utils import txt_read, txt_write
from macadam.conf.path_config import path_root
import json


def thucnews(code_type):
    """
      将baidu_qa_2019数据集转换存储为macadam需要的格式
    """
    path_corpus_text_classification_thucnews = os.path.join(path_root, "data", "corpus",
                                                            "text_classification", "thucnews")
    datas = txt_read(os.path.join(path_corpus_text_classification_thucnews, "{}.txt".format(code_type)))
    train_data = []

    for da in datas:
        da_sp = da.split("\t")
        y = da_sp[0]
        x = da_sp[1]
        # texts2其实是None,但是为了测试模拟, 所以实际取了值
        # xy = {"x":{"text":x, "texts2":[x[0], x[1:3]]}, "y":y}
        xy = {"x": {"text": x.strip(), "texts2": []}, "y": y}
        xy_json = json.dumps(xy, ensure_ascii=False) + "\n"
        train_data.append(xy_json)

        # train_data.append((da_sp[1], da_sp[0]))
    txt_write(train_data, os.path.join(path_corpus_text_classification_thucnews, "{}.json".format(code_type)))

    mm = 0



def baidu_qa_2019(code_type):
    """
      将baidu_qa_2019数据集转换存储为macadam需要的格式
    """
    path_corpus_tc = os.path.join(path_root, "data", "corpus",
                                  "text_classification", "baidu_qa_2019")
    path_real = os.path.join(path_corpus_tc, "{}.csv".format(code_type))
    datas = txt_read(path_real)
    train_data = []

    for da in datas[1:]:
        da_sp = da.split(",")
        y = da_sp[0]
        x = da_sp[1].replace(" ", "")
        # texts2其实是None,但是为了测试模拟, 所以实际取了值
        xy = {"x": {"text": x.strip(), "texts2": []}, "y": y}
        xy_json = json.dumps(xy, ensure_ascii=False) + "\n"
        train_data.append(xy_json)

    txt_write(train_data, os.path.join(path_corpus_tc, "{}.json".format(code_type)))


if __name__ == '__main__':

    for code_type in ["train", "dev", "test"]:

        baidu_qa_2019(code_type)

        # thucnews(code_type)

        mm = 0


