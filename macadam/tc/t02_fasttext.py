# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/5/7 20:43
# @author  : Mo
# @function: FastText  [Bag of Tricks for Efﬁcient Text Classiﬁcation](https://arxiv.org/abs/1607.01759)


from macadam.base.graph import graph
from macadam import K, L, M, O


class FastTextGraph(graph):
    def __init__(self, hyper_parameters):
        super().__init__(hyper_parameters)

    def build_model(self, inputs, outputs):
        x_m = L.GlobalMaxPooling1D()(outputs)
        x_g = L.GlobalAveragePooling1D()(outputs)
        x = L.Concatenate()([x_g, x_m])
        x = L.Dense(min(max(self.label, 128), self.embed_size), activation=self.activate_mid)(x)
        x = L.Dropout(self.dropout)(x)
        self.outputs = L.Dense(units=self.label, activation=self.activate_end)(x)
        self.model = M.Model(inputs=inputs, outputs=self.outputs)
        self.model.summary(132)


# 注意: 随着语料库的增加(word, bi-gram,tri-gram)，内存需求也会不断增加，严重影响模型构建速度:

# 一、自己的思路(macadam, 中文):
# 1. 可以去掉频次高的前后5%的n-gram(, 没有实现)
# 2. 降低embed_size, 从常规的300变为默认64
# 3. 将numpy.array转化时候float32改为默认float16

# 二、其他思路(英文)
# 1. 过滤掉出现次数少的单词
# 2. 使用hash存储
# 3. 由采用字粒度变化为采用词粒度(英文)

