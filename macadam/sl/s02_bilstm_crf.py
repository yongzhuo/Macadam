# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/5/6 21:13
# @author  : Mo
# @function: Bi-LSTM-CRF
# @url     : Bidirectional LSTM-CRF Models for Sequence Tagging(https://arxiv.org/pdf/1508.01991.pdf)


from bert4keras.layers import ConditionalRandomField
from macadam import keras, K, O, C, L, M
from macadam.base.graph import graph


class BiLstmCRFGraph(graph):
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
        self.rnn_type = hyper_parameters["graph"].get("rnn_type", "LSTM") # 1, 2, 3

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
        x = None
        if self.rnn_type == "CUDNNLSTM":
            rnn_cell = L.CuDNNLSTM
        else:
            rnn_cell = L.LSTM
        # Bi-LSTM
        for nrl in range(self.num_rnn_layers):
            x = L.Bidirectional(rnn_cell(units=self.rnn_unit,
                                         return_sequences=True,
                                         activation=self.activate_mid,
                                         ))(outputs)
            x = L.Dropout(self.dropout)(x)
        if self.use_crf:
            x = L.Dense(units=self.label, activation=self.activate_end)(x)
            self.CRF = ConditionalRandomField(self.crf_lr_multiplier, name="crf_bert4keras")
            self.outputs = self.CRF(x)
            self.trans = K.eval(self.CRF.trans).tolist()
            self.loss = self.CRF.dense_loss if self.use_onehot else self.CRF.sparse_loss
            self.metrics = [self.CRF.dense_accuracy if self.use_onehot else self.CRF.sparse_accuracy]
        else:
            self.outputs = L.TimeDistributed(L.Dense(units=self.label, activation=self.activate_end))(x)
        self.model = M.Model(inputs, self.outputs)
        self.model.summary(132)


class BiGruCRFGraph(graph):
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
        self.rnn_type = hyper_parameters["graph"].get("rnn_type", "GRU") # 1, 2, 3

    def build_model(self, inputs, outputs):
        """
        build_model.
        Args:
            inputs: tensor, input of model
            outputs: tensor, output of model
        Returns:
            None
        """
        # CuDNNGRU or GRU
        x = None
        if self.rnn_type.upper() == "CUDNNGRU":
            rnn_cell = L.CuDNNGRU
        else:
            rnn_cell = L.GRU
        # Bi-GRU
        for nrl in range(self.num_rnn_layers):
            x = L.Bidirectional(rnn_cell(units=self.rnn_unit,
                                         return_sequences=True,
                                         activation=self.activate_mid,
                                         ))(outputs)
            x = L.Dropout(self.dropout)(x)
        if self.use_crf:
            x = L.Dense(units=self.label, activation=self.activate_end)(x)
            self.CRF = ConditionalRandomField(self.crf_lr_multiplier, name="crf_bert4keras")
            self.outputs = self.CRF(x)
            self.trans = K.eval(self.CRF.trans).tolist()
            self.loss = self.CRF.dense_loss if self.use_onehot else self.CRF.sparse_loss
            self.metrics = [self.CRF.dense_accuracy if self.use_onehot else self.CRF.sparse_accuracy]
        else:
            self.outputs = L.TimeDistributed(L.Dense(units=self.label, activation=self.activate_end))(x)
        self.model = M.Model(inputs, self.outputs)
        self.model.summary(132)
