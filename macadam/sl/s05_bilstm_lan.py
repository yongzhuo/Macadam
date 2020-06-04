# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/5/6 21:13
# @author  : Mo
# @function: Bi-LSTM-LAN(双向-长短时记忆神经网络 + 注意力机制)
# @url     : Hierarchically-Reﬁned Label Attention Network for Sequence Labeling(https://arxiv.org/abs/1908.08676v2)


from bert4keras.layers import ConditionalRandomField
from macadam.base.layers import SelfAttention
from macadam import keras, K, O, C, L, M
from macadam.base.graph import graph


class BiLstmLANGraph(graph):
    def __init__(self, hyper_parameters):
        """
        Init of hyper_parameters and build_embed.
        Args:
            hyper_parameters: hyper_parameters of all, which contains "sharing", "embed", "graph", "train", "save" and "data".
        Returns:
            None
        """
        super().__init__(hyper_parameters)
        self.num_rnn_layers = hyper_parameters["graph"].get("num_rnn_layers", 2) # 1, 2, 3
        self.crf_lr_multiplier = hyper_parameters.get("train", {}).get("crf_lr_multiplier", 1 if self.embed_type
                                                                        in ["WARD", "RANDOM"] else 3200)

    def build_model(self, inputs, outputs):
        """
        build_model.
        Args:
            inputs: tensor, input of model
            outputs: tensor, output of model
        Returns:
            None
        """
        # LSTM or GRU
        if self.rnn_type == "LSTM":
            rnn_cell = L.LSTM
        elif self.rnn_type == "CuDNNLSTM":
            rnn_cell = L.CuDNNLSTM
        elif self.rnn_type == "CuDNNGRU":
            rnn_cell = L.CuDNNGRU
        else:
            rnn_cell = L.GRU
        # Bi-LSTM-LAN
        for nrl in range(self.num_rnn_layers):
            x = L.Bidirectional(rnn_cell(units=self.rnn_unit*(nrl+1),
                                         return_sequences=True,
                                         activation=self.activate_mid,
                                         ))(outputs)
            x_att = SelfAttention(K.int_shape(x)[-1])(x)
            outputs = L.Concatenate()([x, x_att])
            outputs = L.Dropout(self.dropout)(outputs)
        if self.use_crf:
            x = L.Dense(units=self.label, activation=self.activate_end)(outputs)
            self.CRF = ConditionalRandomField(self.crf_lr_multiplier, name="crf_bert4keras")
            self.outputs = self.CRF(x)
            self.trans = K.eval(self.CRF.trans).tolist()
            self.loss = self.CRF.dense_loss if self.use_onehot else self.CRF.sparse_loss
            self.metrics = [self.CRF.dense_accuracy if self.use_onehot else self.CRF.sparse_accuracy]
        else:
            self.outputs = L.TimeDistributed(L.Dense(units=self.label, activation=self.activate_end))(outputs)
        self.model = M.Model(inputs, self.outputs)
        self.model.summary(132)

