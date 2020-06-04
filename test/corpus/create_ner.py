# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/5/9 23:38
# @author  : Mo
# @function: original corpus of name-entity-recognition change to standardized, 将训练数据语料格式转为macadam需要的格式

from macadam.conf.path_config import path_ner_people_1998
import json
import os


def read_ner_from_column(path):
    """
    1. 读取列状NER数据
    """
    xs, ys = [], []
    with open(path, "r", encoding="utf-8") as fo:
        lines = fo.read().splitlines()
        x, y = [], []
        for line in lines:
            rows = line.split(" ")
            if len(rows) != 2:
                mm = 0
            if len(rows) == 1: # 中间空格部分
                xs.append(x)
                ys.append(y)
                x = []
                y = []
            else:
                x.append(rows[0])
                y.append(rows[1])
    return xs, ys


def ner_columns_to_json(data):
    """
    2. 将两行式NER数据转为macadam需要的json格式
    """
    datas_json = []
    for i in range(len(data[1])):
        line_json = {"x":{"text":None, "texts2":[]}, "y":None}
        x = data[0][i]
        y = data[1][i]
        line_json["x"]["text"] = "".join(x)
        line_json["y"] = y
        line_json_dumps = json.dumps(line_json, ensure_ascii=False) + "\n"
        datas_json.append(line_json_dumps)
    return datas_json


def read_from_columns_and_write_to_macadam(path, path_save):
    """
    即1/2联合, 读取列状的NER数据, 并将其转换存储为macadam需要的格式
    """
    ner_data = read_ner_from_column(path)
    datas_json = ner_columns_to_json(ner_data)
    with open(path_save, "w", encoding="utf-8") as fw:
        fw.writelines(datas_json)
        fw.close()


if __name__ == '__main__':

    for code_type in ["train", "dev", "test"]:
        # code_type = "test"
        path_ner = os.path.join(path_ner_people_1998, "{}.txt".format(code_type))
        path_ner_save = os.path.join(path_ner_people_1998, "{}.json".format(code_type))
        read_from_columns_and_write_to_macadam(path_ner, path_ner_save)

        mm = 0



