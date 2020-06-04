# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/5/7 21:06
# @author  : Mo
# @function: TextRCNN  [Recurrent Convolutional Neural Networks for TextClassiﬁcation](http://www.nlpr.ia.ac.cn/cip/~liukang/liukangPageFile/Recurrent%20Convolutional%20Neural%20Networks%20for%20Text%20Classification.pdf)


from macadam.base.graph import graph
from macadam import keras, K, L, M, O


class RCNNGraph(graph):
    def __init__(self, hyper_parameters):
        self.rnn_layer_repeat = hyper_parameters.get("graph", {}).get("rnn_layer_repeat", 2)
        super().__init__(hyper_parameters)

    def build_model(self, inputs, outputs):
        # rnn type, RNN的类型
        if self.rnn_type == "LSTM":
            layer_cell = L.LSTM
        else:
            layer_cell = L.GRU
        # backword, 反向
        x_backwords = layer_cell(units=self.rnn_unit,
                                 return_sequences=True,
                                 kernel_regularizer=keras.regularizers.l2(0.32 * 0.1),
                                 recurrent_regularizer=keras.regularizers.l2(0.32),
                                 go_backwards=True)(outputs)
        x_backwords_reverse = L.Lambda(lambda x: K.reverse(x, axes=1))(x_backwords)
        # fordword, 前向
        x_fordwords = layer_cell(units=self.rnn_unit,
                                 return_sequences=True,
                                 kernel_regularizer=keras.regularizers.l2(0.32 * 0.1),
                                 recurrent_regularizer=keras.regularizers.l2(0.32),
                                 go_backwards=False)(outputs)
        # concatenate, 拼接
        x_feb = L.Concatenate(axis=2)([x_fordwords, outputs, x_backwords_reverse])
        # dropout, 随机失活
        x_feb = L.Dropout(self.dropout)(x_feb)
        # Concatenate, 拼接后的embedding_size
        dim_2 = K.int_shape(x_feb)[2]
        x_feb_reshape = L.Reshape((self.length_max, dim_2, 1))(x_feb)
        # n-gram, conv, maxpool, 使用n-gram进行卷积和池化
        conv_pools = []
        for filter in self.filters_size:
            conv = L.Conv2D(filters=self.filters_num,
                            kernel_size=(filter, dim_2),
                            padding='valid',
                            kernel_initializer='normal',
                            activation='relu',
                            )(x_feb_reshape)
            pooled = L.MaxPooling2D(pool_size=(self.length_max - filter + 1, 1),
                                    strides=(1, 1),
                                    padding='valid',
                                    )(conv)
            conv_pools.append(pooled)
        # concatenate, 拼接TextCNN
        x = L.Concatenate()(conv_pools)
        x = L.Dropout(self.dropout)(x)
        # dense-mid, 中间全连接到中间的隐藏元
        x = L.Flatten()(x)
        x = L.Dense(units=min(max(self.label, 64), self.embed_size), activation=self.activate_mid)(x)
        x = L.Dropout(self.dropout)(x)
        # dense-end, 最后一层, dense到label
        self.outputs = L.Dense(units=self.label, activation=self.activate_end)(x)
        self.model = M.Model(inputs=inputs, outputs=self.outputs)
        self.model.summary(132)
