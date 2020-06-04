# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/4/30 21:12
# @author  : Mo
# @function: CRF(crf of bert4keras, 条件概率随机场)
# @url     : Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data(https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers)


from bert4keras.layers import ConditionalRandomField
from macadam import keras, K, L, M, O
from macadam.base.graph import graph


class CRFGraph(graph):
    def __init__(self, hyper_parameters):
        """
        Init of hyper_parameters and build_embed.
        Args:
            hyper_parameters: hyper_parameters of all, which contains "sharing", "embed", "graph", "train", "save" and "data".
        Returns:
            None
        """
        super().__init__(hyper_parameters)
        self.crf_lr_multiplier = hyper_parameters.get("train", {}).get("crf_lr_multiplier",
                                        1 if self.embed_type in ["WARD", "RANDOM"] else 3200)

    def build_model(self, inputs, outputs):
        x = L.Dense(units=self.label, activation=self.activate_mid)(outputs)
        self.CRF = ConditionalRandomField(self.crf_lr_multiplier, name="crf_bert4keras")
        self.outputs = self.CRF(x)
        self.model = M.Model(inputs, self.outputs)
        self.model.summary(132)
        self.trans = K.eval(self.CRF.trans).tolist()
        self.loss = self.CRF.dense_loss if self.use_onehot else self.CRF.sparse_loss
        self.metrics = [self.CRF.dense_accuracy if self.use_onehot else self.CRF.sparse_accuracy]
