# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/5/12 22:46
# @author  : Mo
# @function: Bi-LSTM-CNNs-CRF, 有改动, 为更适普的"卷积神经网络+循环神经网络", CNN + LSTM
# @url:      Bidirectional LSTM-CRF Models for Sequence Tagging(https://arxiv.org/pdf/1508.01991.pdf)


from bert4keras.layers import ConditionalRandomField
from macadam import keras, K, O, C, L, M
from macadam.base.graph import graph


class CnnLstmGraph(graph):
    def __init__(self, hyper_parameters):
        """
        Init of hyper_parameters and build_embed.
        Args:
            hyper_parameters: hyper_parameters of all, which contains "sharing", "embed", "graph", "train", "save" and "data".
        Returns:
            None
        """
        super().__init__(hyper_parameters)
        self.num_rnn_layers = hyper_parameters["graph"].get("num_rnn_layers", 1) # 1, 2, 3
        self.crf_lr_multiplier = hyper_parameters.get("train", {}).get("crf_lr_multiplier",
                                        1 if self.embed_type in ["WARD", "RANDOM"] else 3200)

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
        # CNN-LSTM, 提取n-gram特征和最大池化， 一般不用平均池化
        conv_pools = []
        for i in range(len(self.filters_size)):
            conv = L.Conv1D(name="conv-{0}-{1}".format(i, self.filters_size[i]),
                            kernel_size=self.filters_size[i],
                            activation=self.activate_mid,
                            filters=self.filters_num,
                            padding='same',
                            )(outputs)
            conv_rnn = L.Bidirectional(rnn_cell(name="bi-lstm-{0}-{1}".format(i, self.filters_size[i]),
                                                activation=self.activate_mid,
                                                return_sequences=True,
                                                units=self.rnn_unit,)
                                       )(conv)
            x_dropout = L.Dropout(rate=self.dropout, name="dropout-{0}-{1}".format(i, self.filters_size[i]))(conv_rnn)
            conv_pools.append(x_dropout)
        # 拼接
        x = L.Concatenate(axis=-1)(conv_pools)
        x = L.Dropout(self.dropout)(x)
        # CRF or Dense
        if self.use_crf:
            x = L.Dense(units=self.label, activation=self.activate_end)(x)
            self.CRF = ConditionalRandomField(self.crf_lr_multiplier, name="crf_bert4keras")
            self.outputs = self.CRF(x)
            self.trans = K.eval(self.CRF.trans).tolist()
            self.loss = self.CRF.dense_loss if self.use_onehot else self.CRF.sparse_loss
            self.metrics = [self.CRF.dense_accuracy if self.use_onehot else self.CRF.sparse_accuracy]
        else:
            self.outputs = L.TimeDistributed(L.Dense(units=self.label, activation=self.activate_end, name="dense-output"))(x)
        self.model = M.Model(inputs, self.outputs)
        self.model.summary(132)
