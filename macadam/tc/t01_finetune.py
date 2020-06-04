# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/4/30 16:26
# @author  : Mo
# @function: graph of finetune of bert,robert,albert,xlnet


from macadam.base.graph import graph
from macadam import keras, K, L, M, O


class FineTuneGraph(graph):
    def __init__(self, hyper_parameters):
        super().__init__(hyper_parameters)

    def build_model(self, inputs, outputs):
        if self.embed_type in ["xlnet"]:
            # x = L.Lambda(lambda x: x[:, -2:-1, :])(outputs)  # xlnet获取CLS
            x = L.Lambda(lambda x: x[:, -1], name="Token-CLS")(outputs)
        else:
            # x = L.Lambda(lambda x: x[:, 0:1, :])(outputs)  # bert-like获取CLS
            x = L.Lambda(lambda x: x[:, 0], name="Token-CLS")(outputs)
        # x = L.Flatten()(x)
        # 最后就是softmax
        self.outputs = L.Dense(self.label, activation=self.activate_end,
                               kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02))(x)
        self.model = M.Model(inputs, self.outputs)
        self.model.summary(132)
