# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/5/8 0:47
# @author  : Mo
# @function: self-attention


from macadam.base.layers import SelfAttention
from macadam.base.graph import graph
from macadam import keras, K, L, M


class SelfAttentionGraph(graph):
    def __init__(self, hyper_parameters):
        self.dropout_spatial = hyper_parameters.get("graph", {}).get('dropout_spatial', 0.2)
        super().__init__(hyper_parameters)

    def build_model(self, inputs, outputs):
        x = L.SpatialDropout1D(self.dropout_spatial)(outputs)
        x = SelfAttention(K.int_shape(outputs)[-1])(x)
        x_max = L.GlobalMaxPooling1D()(x)
        x_avg = L.GlobalAveragePooling1D()(x)
        x = L.Concatenate()([x_max, x_avg])
        x = L.Dropout(self.dropout)(x)
        x = L.Flatten()(x)
        # dense-mid
        x = L.Dense(units=min(max(self.label, 64), self.embed_size), activation=self.activate_mid)(x)
        x = L.Dropout(self.dropout)(x)
        # dense-end, 最后一层, dense到label
        self.outputs = L.Dense(units=self.label, activation=self.activate_end)(x)
        self.model = M.Model(inputs=inputs, outputs=self.outputs)
        self.model.summary(132)
