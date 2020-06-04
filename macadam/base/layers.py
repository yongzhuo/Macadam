# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/4/26 22:00
# @author  : Mo
# @function: layers of keras


from __future__ import print_function, division
from macadam import keras, K, L, M, O


__all__ = ["AttentionWeightedAverage",
           "dynamic_k_max_pooling",
           "wide_convolution",
           "NonMaskingLayer",
           "prem_fold",
           "select_k",
           ]


### BERT ################################
class NonMaskingLayer(L.Layer):
    """
    fix convolutional 1D can"t receive masked input, detail: https://github.com/keras-team/keras/issues/4978
    thanks for https://github.com/jacoxu
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMaskingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x

    def get_output_shape_for(self, input_shape):
        return input_shape


### DCNN ################################
class wide_convolution(L.Layer):
    """
        paper: http://www.aclweb.org/anthology/P14-1062
        paper title: "A Convolutional Neural Network for Modelling Sentences"
        宽卷积, 如果s表示句子最大长度, m为卷积核尺寸,
           则宽卷积输出为 s + m − 1,
           普通卷积输出为 s - m + 1.
        github keras实现可以参考: https://github.com/AlexYangLi/TextClassification/blob/master/models/keras_dcnn_model.py
    """
    def __init__(self, filter_num=300, filter_size=3, **kwargs):
        self.filter_size = filter_size
        self.filter_num = filter_num
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        x_input_pad = L.ZeroPadding1D((self.filter_size-1, self.filter_size-1))(inputs)
        conv_1d = L.Conv1D(filters=self.filter_num,
                         kernel_size=self.filter_size,
                         strides=1,
                         padding="VALID",
                         kernel_initializer="normal", # )(x_input_pad)
                         activation="tanh")(x_input_pad)
        return conv_1d

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1] + self.filter_size - 1, input_shape[-1]

    def get_config(self):
        config = {"filter_size":self.filter_size,
                  "filter_num": self.filter_num}
        base_config = super(wide_convolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
class dynamic_k_max_pooling(L.Layer):
    """
        paper:        http://www.aclweb.org/anthology/P14-1062
        paper title:  A Convolutional Neural Network for Modelling Sentences
        Reference:    https://stackoverflow.com/questions/51299181/how-to-implement-k-max-pooling-in-tensorflow-or-keras
        动态K-max pooling
            k的选择为 k = max(k, s * (L-1) / L)
            其中k为预先选定的设置的最大的K个值，s为文本最大长度，L为第几个卷积层的深度（单个卷积到连接层等）
        github tf实现可以参考: https://github.com/lpty/classifier/blob/master/a04_dcnn/model.py
    """
    def __init__(self, top_k=3, **kwargs):
        self.top_k = top_k
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        import tensorflow as tf
        inputs_reshape = tf.transpose(inputs, perm=[0, 2, 1])
        pool_top_k = tf.nn.top_k(input=inputs_reshape, k=self.top_k, sorted=False).values
        pool_top_k_reshape = tf.transpose(pool_top_k, perm=[0, 2, 1])
        return pool_top_k_reshape

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.top_k, input_shape[-1]

    def get_config(self):
        config = {"top_k": self.top_k}
        base_config = super(dynamic_k_max_pooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
class prem_fold(L.Layer):
    """
        paper:       http://www.aclweb.org/anthology/P14-1062
        paper title: A Convolutional Neural Network for Modelling Sentences
        detail:      垂直于句子长度的方向，相邻值相加，就是embedding层300那里，（0+1,2+3...298+299）
        github tf实现可以参考: https://github.com/lpty/classifier/blob/master/a04_dcnn/model.py
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, conv_shape):
        super().build(conv_shape)

    def call(self, convs):
        conv1 = convs[:, :, ::2]
        conv2 = convs[:, :, 1::2]
        conv_fold = L.Add()([conv1, conv2])
        return conv_fold

    def compute_output_shape(self, conv_shape):
        return conv_shape[0], conv_shape[1], int(conv_shape[2] / 2)

    def get_config(self):
        base_config = super(prem_fold, self).get_config()
        return dict(list(base_config.items()))
def select_k(len_max, length_conv, length_curr, k_con=3):
    """
        dynamic k max pooling中的k获取
    :param len_max:int, max length of input sentence 
    :param length_conv: int, deepth of all convolution layer
    :param length_curr: int, deepth of current convolution layer
    :param k_con: int, k of constant 
    :return: int, return 
    """
    if type(len_max) != int:
        len_max = len_max[0]
    if type(length_conv) != int:
        length_conv = length_conv[0]
    if length_conv >= length_curr:
        k_ml = int(len_max * (length_conv-length_curr) / length_conv)
        k = max(k_ml, k_con)
    else:
        k = k_con
    return k


### DeepMoji ############################
class AttentionWeightedAverage(L.Layer):
    """
    codes from: https://github.com/bfelbo/DeepMoji/blob/master/deepmoji/attlayer.py
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = keras.initializers.get("uniform")
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [L.InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name="{}_W".format(self.name),
                                 initializer=self.init)
        # self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses "max trick" for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None

    def get_config(self):
        config = {"return_attention": self.return_attention,}
        base_config = super(AttentionWeightedAverage, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


### SelfAttention #######################
class SelfAttention(L.Layer):
    """
        self attention,
        codes from:  https://mp.weixin.qq.com/s/qmJnyFMkXVjYBwoR_AQLVA
    """
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def build(self, input_shape):
        # W、K and V
        self.kernel = self.add_weight(name='WKV',
                                      shape=(3, input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      regularizer=keras.regularizers.l1_l2(0.0000032),
                                      trainable=True)
        super().build(input_shape)

    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])
        # print("WQ.shape",WQ.shape)
        # print("K.permute_dimensions(WK, [0, 2, 1]).shape",K.permute_dimensions(WK, [0, 2, 1]).shape)
        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))
        QK = QK / (64**0.5)
        QK = K.softmax(QK)
        # print("QK.shape",QK.shape)
        V = K.batch_dot(QK, WV)
        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        config = {"output_dim": self.output_dim,}
        base_config = super(SelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


### Capsule #############################
def squash_bojone(x, axis=-1):
    """
       activation of squash
    :param x: vector
    :param axis: int
    :return: vector
    """
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale
class Capsule_bojone(L.Layer):
    """
        # auther: bojone
        # explain: A Capsule Implement with Pure Keras
        # github: https://github.com/bojone/Capsule/blob/master/Capsule_Keras.py
    """
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1),
                 share_weights=True, activation='default', **kwargs):
        super(Capsule_bojone, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash_bojone
        else:
            self.activation = L.Activation(activation)

    def build(self, input_shape):
        super(Capsule_bojone, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        outputs = None
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

    def get_config(self):
        config = {"num_capsule": self.num_capsule,
                  "dim_capsule": self.dim_capsule,
                  "routings": self.routings,
                  "kernel_size": self.kernel_size,
                  "share_weights": self.share_weights,
                  # "activation": self.activation
                  }
        base_config = super(Capsule_bojone, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# macadam.custom_objects["NonMaskingLayer"] = NonMaskingLayer
# macadam.custom_objects["dynamic_k_max_pooling"] = dynamic_k_max_pooling
# macadam.custom_objects["wide_convolution"] = wide_convolution
# macadam.custom_objects["prem_fold"] = prem_fold
# macadam.custom_objects["AttentionWeightedAverage"] = AttentionWeightedAverage
# macadam.custom_objects["SelfAttention"] = SelfAttention

custom_objects_macadam = {"NonMaskingLayer": NonMaskingLayer,
                          "dynamic_k_max_pooling": dynamic_k_max_pooling,
                          "wide_convolution": wide_convolution,
                          "prem_fold": prem_fold,
                          "AttentionWeightedAverage": AttentionWeightedAverage,
                          "SelfAttention": SelfAttention,
                          "Capsule_bojone": Capsule_bojone}

keras.utils.get_custom_objects().update(custom_objects_macadam)

