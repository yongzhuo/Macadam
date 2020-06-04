# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/5/7 21:06
# @author  : Mo
# @function: CharCNN  [Character-level Convolutional Networks for Text Classiﬁcation](https://arxiv.org/pdf/1509.01626.pdf)


from macadam.base.graph import graph
from macadam import K, L, M, O


class CharCNNGraph(graph):
    def __init__(self, hyper_parameters):
        self.char_cnn_layers = hyper_parameters.get("graph", {}).get('char_cnn_layers', [[256, 7, 3], [256, 7, 3],
                                                                 [256, 3, -1], [256, 3, -1], [256, 3, -1], [256, 3, 3]], )
        self.full_connect_layers = hyper_parameters.get("graph", {}).get('full_connect_layers', [1024, 1024], )
        self.threshold = hyper_parameters.get("graph", {}).get('threshold', 1e-6)
        super().__init__(hyper_parameters)

    def build_model(self, inputs, outputs):
        x = None
        # cnn + pool
        for char_cnn_size in self.char_cnn_layers:
            x = L.Convolution1D(filters=char_cnn_size[0],
                              kernel_size=char_cnn_size[1], )(outputs)
            x = L.ThresholdedReLU(self.threshold)(x)
            if char_cnn_size[2] != -1:
                x = L.MaxPooling1D(pool_size=char_cnn_size[2], strides=1)(x)
        x = L.Flatten()(x)
        # full-connect 2
        for full in self.full_connect_layers:
            x = L.Dense(units=full, )(x)
            x = L.ThresholdedReLU(self.threshold)(x)
            x = L.Dropout(self.dropout)(x)
        # dense label
        self.outputs = L.Dense(units=self.label, activation=self.activate_end)(x)
        self.model = M.Model(inputs=inputs, outputs=self.outputs)
        self.model.summary(132)
