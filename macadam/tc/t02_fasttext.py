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

