# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/5/12 22:46
# @author  : Mo
# @function: DGCNN(Dilate Gated Convolutional Neural Network, 即"膨胀门卷积神经网络", IDCNN + CRF)
# @url     : Multi-Scale Context Aggregation by Dilated Convolutions(https://arxiv.org/abs/1511.07122)


from bert4keras.layers import ConditionalRandomField
from macadam import keras, K, O, C, L, M
from macadam.base.graph import graph


class DGCNNGraph(graph):
    def __init__(self, hyper_parameters):
        """
        Init of hyper_parameters and build_embed.
        Args:
            hyper_parameters: hyper_parameters of all, which contains "sharing", "embed", "graph", "train", "save" and "data".
        Returns:
            None
        """
        super().__init__(hyper_parameters)
        self.atrous_rates = hyper_parameters["graph"].get("atrous_rates", [2, 1, 2]) # 1, 2, 3
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
        # CNN, 提取n-gram特征和最大池化, DGCNN膨胀卷积(IDCNN)
        conv_pools = []
        for i in range(len(self.filters_size)):
            conv = L.Conv1D(name="conv-{0}-{1}".format(i, self.filters_size[i]),
                            dilation_rate=self.atrous_rates[0],
                            kernel_size=self.filters_size[i],
                            activation=self.activate_mid,
                            filters=self.filters_num,
                            padding="SAME",
                            )(outputs)
            for j in range(len(self.atrous_rates) - 1):
                conv = L.Conv1D(name="conv-{0}-{1}-{2}".format(i, self.filters_size[i], j),
                                dilation_rate=self.atrous_rates[j],
                                kernel_size=self.filters_size[i],
                                activation=self.activate_mid,
                                filters=self.filters_num,
                                padding="SAME",
                                )(conv)
                conv = L.Dropout(name="dropout-{0}-{1}-{2}".format(i, self.filters_size[i], j),
                                 rate=self.dropout,)(conv)
            conv_pools.append(conv)
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
            x = L.Bidirectional(L.GRU(activation=self.activate_mid,
                                      return_sequences=True,
                                      units=self.rnn_unit,
                                      name="bi-gru",)
                                )(x)
            self.outputs = L.TimeDistributed(L.Dense(activation=self.activate_end,
                                                     name="dense-output",
                                                     units=self.label,))(x)
        self.model = M.Model(inputs, self.outputs)
        self.model.summary(132)
