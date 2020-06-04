# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/5/6 21:13
# @author  : Mo
# @function: LATTICE-LSTM-BATCH(双向-长短时记忆神经网络)
# @url     : An Encoding Strategy Based Word-Character LSTM for Chinese NER(https://www.aclweb.org/anthology/N19-1247/)


from bert4keras.layers import ConditionalRandomField
from macadam.base.layers import SelfAttention
from macadam import keras, K, O, C, L, M
from macadam.base.graph import graph


class LatticeLSTMBatchgraph(graph):
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
        print(self.crf_lr_multiplier)
        self.wclstm_embed_type = hyper_parameters["graph"].get("wclstm_embed_type", "SHORT")  # "ATTENTION", "SHORT", "LONG", "CNN"

    def build_model(self, inputs, outputs):
        """
        build_model.
        Args:
            inputs: tensor, input of model
            outputs: tensor, output of model
        Returns:
            None
        """
        embed_char = outputs[0]
        embed_word = outputs[1]
        if self.wclstm_embed_type == "ATTNENTION":
            x_word = L.TimeDistributed(SelfAttention(K.int_shape(embed_word)[-1]))(embed_word)
            x_word_shape = K.int_shape(x_word)
            x_word = L.Reshape(target_shape=(x_word_shape[:2], x_word_shape[2]*x_word_shape[3]))
            x_word = L.Dense(self.embed_size, activation=self.activate_mid)(x_word)
        # elif self.wclstm_embed_type == "SHORT":
        else:
            x_word = L.Lambda(lambda x: x[:, :, 0, :])(embed_word)
        outputs_concat = L.Concatenate(axis=-1)([embed_char, x_word])
        # LSTM or GRU
        if self.rnn_type == "LSTM":
            rnn_cell = L.LSTM
        elif self.rnn_type == "CuDNNLSTM":
            rnn_cell = L.CuDNNLSTM
        elif self.rnn_type == "CuDNNGRU":
            rnn_cell = L.CuDNNGRU
        else:
            rnn_cell = L.GRU
        # Bi-LSTM-CRF
        for nrl in range(self.num_rnn_layers):
            x = L.Bidirectional(rnn_cell(units=self.rnn_unit*(nrl+1),
                                         return_sequences=True,
                                         activation=self.activate_mid,
                                         ))(outputs_concat)
            outputs = L.Dropout(self.dropout)(x)
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
