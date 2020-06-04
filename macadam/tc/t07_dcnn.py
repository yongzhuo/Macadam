# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/5/7 21:49
# @author  : Mo
# @function: DCNN  [A Convolutional Neural Network for Modelling Sentences](http://www.aclweb.org/anthology/P14-1062)


from macadam.base.layers import wide_convolution, dynamic_k_max_pooling, prem_fold, select_k
from macadam.base.graph import graph
from macadam import keras, K, L, M, O


class DCNNGraph(graph):
    def __init__(self, hyper_parameters):
        super().__init__(hyper_parameters)
        self.filters_size = [[10, 7, 5], [6, 4, 3]]  # 三层时候

    def build_model(self, inputs, outputs):
        # rnn type, RNN的类型
        pools = []
        for i in range(len(self.filters_size)):
            # 第一个, 宽卷积, 动态k-max池化
            conv_1 = wide_convolution(name="wide_convolution_{}".format(i),
                                      filter_num=self.filters_num, filter_size=self.filters_size[i][0])(outputs)
            top_k_1 = select_k(self.length_max, len(self.filters_size[i]), 1)  # 求取k
            dynamic_k_max_pooled_1 = dynamic_k_max_pooling(top_k=top_k_1)(conv_1)
            # 第二个, 宽卷积, 动态k-max池化
            conv_2 = wide_convolution(name="wide_convolution_{}_{}".format(i, i),
                                      filter_num=self.filters_num, filter_size=self.filters_size[i][1])(
                dynamic_k_max_pooled_1)
            top_k_2 = select_k(self.length_max, len(self.filters_size[i]), 2)
            dynamic_k_max_pooled_2 = dynamic_k_max_pooling(top_k=top_k_2)(conv_2)
            # 第三层, 宽卷积, Fold层, 动态k-max池化
            conv_3 = wide_convolution(name="wide_convolution_{}_{}_{}".format(i, i, i), filter_num=self.filters_num,
                                      filter_size=self.filters_size[i][2])(dynamic_k_max_pooled_2)
            fold_conv_3 = prem_fold()(conv_3)
            top_k_3 = select_k(self.length_max, len(self.filters_size[i]), 3)  # 求取k
            dynamic_k_max_pooled_3 = dynamic_k_max_pooling(top_k=top_k_3)(fold_conv_3)
            pools.append(dynamic_k_max_pooled_3)
        pools_concat = L.Concatenate(axis=1)(pools)
        pools_concat_dropout = L.Dropout(self.dropout)(pools_concat)
        x = L.Flatten()(pools_concat_dropout)
        # dense-end, 最后一层, dense到label
        self.outputs = L.Dense(units=self.label, activation=self.activate_end)(x)
        self.model = M.Model(inputs=inputs, outputs=self.outputs)
        self.model.summary(132)

