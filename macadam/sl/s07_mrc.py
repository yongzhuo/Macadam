# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/5/6 21:13
# @author  : Mo
# @function: MRC(阅读理解式序列标注)
# @url     : A Uniﬁed MRC Framework for Named Entity Recognition(https://arxiv.org/abs/1910.11476)


from bert4keras.layers import ConditionalRandomField
from macadam.base.layers import SelfAttention
from macadam import keras, K, O, C, L, M
from macadam.base.graph import graph


class Mrcgraph(graph):
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
        # todo
        mm = 0
