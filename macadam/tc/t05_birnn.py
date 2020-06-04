# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/5/7 21:06
# @author  : Mo
# @function: BiRNN(BiLSTM, BiGRU)


from macadam.base.graph import graph
from macadam import keras, K, L, M, O


class BiRNNGraph(graph):
    def __init__(self, hyper_parameters):
        self.rnn_layer_repeat = hyper_parameters.get("graph", {}).get("rnn_layer_repeat", 1)
        super().__init__(hyper_parameters)

    def build_model(self, inputs, outputs):
        x = None
        if self.rnn_type == "LSTM":
            rnn_cell = L.LSTM
        elif self.rnn_type == "CuDNNLSTM":
            rnn_cell = L.CuDNNLSTM
        elif self.rnn_type == "CuDNNGRU":
            rnn_cell = L.CuDNNGRU
        else:
            rnn_cell = L.GRU
        # Bi-RNN(LSTM/GRU)
        for _ in range(self.rnn_layer_repeat):
            x = L.Bidirectional(rnn_cell(units=self.rnn_unit,
                                         return_sequences=True,
                                         activation=self.activate_mid
                                         ))(outputs)
            x = L.Dropout(self.dropout)(x)
        # dense-mid
        x = L.Flatten()(x)
        x = L.Dense(units=min(max(self.label, 128), self.embed_size), activation=self.activate_mid)(x)
        x = L.Dropout(self.dropout)(x)
        # dense-end
        self.outputs = L.Dense(units=self.label, activation=self.activate_end)(x)
        self.model = M.Model(inputs=inputs, outputs=self.outputs)
        self.model.summary(132)

