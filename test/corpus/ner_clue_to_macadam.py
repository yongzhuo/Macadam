# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/5/14 21:09
# @author  : Mo
# @function: 将json格式的corpus转化为macadam的格式(clue数据集)


from macadam.base.utils import txt_write, txt_read, save_json, load_json
from macadam.conf.path_config import path_ner_clue_2020
import json
import os


for code_type in ["train", "dev"]:
    # code_type = "test"  # "train", "dev", "test"
    path_train = os.path.join(path_ner_clue_2020, f"{code_type}.json")
    path_save = os.path.join(path_ner_clue_2020, f"ner_clue_2020.{code_type}")
    # path_dev = os.path.join(path_ner_clue_2020, "dev.json")
    # path_tet = os.path.join(path_ner_clue_2020, "tet.json")

    data_train = txt_read(path_train)
    res = []
    for data_line in data_train:
        data_json_save = {"x":{"text":"", "texts2":[]}, "y":[]}
        data_line_json = json.loads(data_line.strip())
        text = data_line_json.get("text")
        label = data_line_json.get("label")

        y = ["O"] * len(text)
        data_json_save["x"]["text"] = text
        for k, v in label.items():
            for k2,v2 in v.items():
                for v2_idx in v2:
                    start = v2_idx[0]
                    end = v2_idx[1]
                    if start==end:
                        y[start] = "S-{}".format(k)
                    ####  BMES标注法  ###
                    # else:
                    #     y[start:end] = ["M-{}".format(k)] * len(k2)
                    #     y[start] = "B-{}".format(k)
                    #     y[end] = "E-{}".format(k)

                    ####  BIO标注法  ###
                    else:
                        y[start:end] = ["I-{}".format(k)] * len(k2)
                        y[start] = "B-{}".format(k)
        data_json_save["y"] = y
        # res.append(data_json_save)
        line_save = json.dumps(data_json_save, ensure_ascii=False) + "\n"
        res.append(line_save)

    txt_write(res, path_save)
    # save_json(res, path_save, indent=4)

mm = 0


# CLUENER 细粒度命名实体识别
#
# 数据分为10个标签类别，分别为:
# 地址（address），
# 书名（book），
# 公司（company），
# 游戏（game），
# 政府（goverment），
# 电影（movie），
# 姓名（name），
# 组织机构（organization），
# 职位（position），
# 景点（scene）

