# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/5/8 0:27
# @author  : Mo
# @function: CRNN  [A C-LSTM Neural Network for Text Classification](https://arxiv.org/abs/1511.08630)


from macadam.base.graph import graph
from macadam import keras, K, L, M, O


class CRNNGraph(graph):
    def __init__(self, hyper_parameters):
        self.dropout_spatial = hyper_parameters.get("graph", {}).get('dropout_spatial', 0.2)
        self.l2 = hyper_parameters.get("graph", {}).get('l2', 0.001)
        super().__init__(hyper_parameters)

    def build_model(self, inputs, outputs):
        # rnn type, RNN的类型
        if self.rnn_unit == "LSTM":
            layer_cell = L.LSTM
        elif self.rnn_unit == "CuDNNLSTM":
            layer_cell = L.CuDNNLSTM
        elif self.rnn_unit == "CuDNNGRU":
            layer_cell = L.CuDNNGRU
        else:
            layer_cell = L.GRU
        # embedding遮挡
        embedding_output_spatial = L.SpatialDropout1D(self.dropout_spatial)(outputs)
        # CNN
        convs = []
        for kernel_size in self.filters_size:
            conv = L.Conv1D(self.filters_num,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='SAME',
                          kernel_regularizer=keras.regularizers.l2(self.l2),
                          bias_regularizer=keras.regularizers.l2(self.l2),
                          )(embedding_output_spatial)
            convs.append(conv)
        x = L.Concatenate(axis=1)(convs)
        # Bi-LSTM, 论文中使用的是LSTM
        x = L.Bidirectional(layer_cell(units=self.rnn_unit,
                                     return_sequences=True,
                                     activation='relu',
                                     kernel_regularizer=keras.regularizers.l2(self.l2),
                                     recurrent_regularizer=keras.regularizers.l2(self.l2)
                                     ))(x)
        x = L.Dropout(self.dropout)(x)
        x = L.Flatten()(x)
        # dense-mid
        x = L.Dense(units=min(max(self.label, 64), self.embed_size), activation=self.activate_mid)(x)
        x = L.Dropout(self.dropout)(x)
        # dense-end, 最后一层, dense到label
        self.outputs = L.Dense(units=self.label, activation=self.activate_end)(x)
        self.model = M.Model(inputs=inputs, outputs=self.outputs)
        self.model.summary(132)

