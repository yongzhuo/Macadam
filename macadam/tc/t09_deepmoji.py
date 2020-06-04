# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/5/8 0:33
# @author  : Mo
# @function: DeepMoji  [Using millions of emoji occurrences to learn any-domain representations for detecting sentiment, emotion and sarcasm](https://arxiv.org/abs/1708.00524)


from macadam.base.layers import AttentionWeightedAverage
from macadam.base.graph import graph
from macadam import keras, K, L, M


class DeepMojiGraph(graph):
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

        x = L.Activation(self.activate_mid)(outputs)
        # embedding遮挡
        x = L.SpatialDropout1D(self.dropout_spatial)(x)

        lstm_0_output = L.Bidirectional(layer_cell(units=self.rnn_unit,
                                                 return_sequences=True,
                                                 activation='relu',
                                                 kernel_regularizer=keras.regularizers.l2(self.l2),
                                                 recurrent_regularizer=keras.regularizers.l2(self.l2)
                                                 ), name="bi_lstm_0")(x)
        lstm_1_output = L.Bidirectional(layer_cell(units=self.rnn_unit,
                                                 return_sequences=True,
                                                 activation='relu',
                                                 kernel_regularizer=keras.regularizers.l2(self.l2),
                                                 recurrent_regularizer=keras.regularizers.l2(self.l2)
                                                 ), name="bi_lstm_1")(lstm_0_output)
        x = L.Concatenate()([lstm_1_output, lstm_0_output, x])
        x = AttentionWeightedAverage(name='attlayer', return_attention=False)(x)
        x = L.Dropout(self.dropout)(x)
        x = L.Flatten()(x)
        # dense-mid
        x = L.Dense(units=min(max(self.label, 64), self.embed_size), activation=self.activate_mid)(x)
        x = L.Dropout(self.dropout)(x)
        # dense-end, 最后一层, dense到label
        self.outputs = L.Dense(units=self.label, activation=self.activate_end)(x)
        self.model = M.Model(inputs=inputs, outputs=self.outputs)
        self.model.summary(132)

