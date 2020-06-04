# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/4/30 16:26
# @author  : Mo
# @function: TextCNN  [Convolutional Neural Networks for Sentence Classiﬁcation](https://arxiv.org/abs/1408.5882)


from macadam.base.graph import graph
from macadam import K, L, M, O


class TextCNNGraph(graph):
    def __init__(self, hyper_parameters):
        super().__init__(hyper_parameters)

    def build_model(self, inputs, outputs):
        embedding_reshape = L.Reshape((self.length_max, self.embed_size, 1))(outputs)
        # 提取n-gram特征和最大池化， 一般不用平均池化
        conv_pools = []
        for filter in self.filters_size:
            conv = L.Conv2D(filters=self.filters_num,
                            kernel_size=(filter, self.embed_size),
                            padding='valid',
                            kernel_initializer='normal',
                            activation='tanh',
                            )(embedding_reshape)
            pooled = L.MaxPool2D(pool_size=(self.length_max - filter + 1, 1),
                                 strides=(1, 1),
                                 padding='valid',
                                 )(conv)
            conv_pools.append(pooled)
        # 拼接
        x = L.Concatenate(axis=-1)(conv_pools)
        x = L.Dropout(self.dropout)(x)
        x = L.Flatten()(x)
        self.outputs = L.Dense(units=self.label, activation=self.activate_end)(x)
        self.model = M.Model(inputs=inputs, outputs=self.outputs)
        self.model.summary(132)
