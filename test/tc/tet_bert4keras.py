# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/5/8 21:22
# @author  : Mo
# @function:


from macadam.base.layers import NonMaskingLayer
from macadam.conf.path_config import path_tc_baidu_qa_2019, path_tc_thucnews
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator, open
from bert4keras.models import build_transformer_model
from bert4keras.backend import keras, set_gelu
from bert4keras.tokenizers import Tokenizer
from keras.layers import Lambda, Dense
import numpy as np
import os


set_gelu('tanh')  # 切换gelu版本

num_classes = 2
maxlen = 31
batch_size = 32

# bert-embedding地址, 必传
path_embed_bert = "D:/soft_install/dataset/bert-model/chinese_L-12_H-768_A-12/"
config_path = path_embed_bert + "bert_config.json"
checkpoint_path = path_embed_bert + "bert_model.ckpt"
dict_path =  path_embed_bert + "vocab.txt"

def load_data(filename):
    D = []
    # count = 1000
    with open(filename, encoding='utf-8') as f:
        l2i = {}
        for l in f:
            label, text = l.strip().split(',')
            if label not in l2i:
                l2i[label] = len(l2i)
                label = l2i[label]
            else:
                label = l2i[label]
            D.append((text.replace(" ", ""), label))
    return D, l2i

# 训练/验证数据地址
path_train = os.path.join(path_tc_baidu_qa_2019, "train.csv")
path_dev = os.path.join(path_tc_baidu_qa_2019, "dev.csv")
# 加载数据集
train_data, l2i = load_data(path_train)
valid_data, l2i2 = load_data(path_dev)


num_classes = len(l2i)


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='bert',
    return_keras_model=True,
)
# 获取bert4keras层输出
num_hidden_layers = 12
output_layers = [bert.get_layer('Transformer-{0}-FeedForward-Norm'.format(i)).output for i in range(num_hidden_layers)]
# 获取实际的layer
features_layers = [output_layers[li] for li in [-1]]
# 输出layer层的merge方式
embedding_layer = features_layers[0]
output_layer = NonMaskingLayer()(embedding_layer)

output = Lambda(lambda x: x[:, 0], name='CLS-token')(output_layer)
output = Dense(
    units=num_classes,
    activation='softmax',
    # kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.input, output)
model.summary()

# 派生为带分段线性学习率的优化器。
# 其中name参数可选，但最好填入，以区分不同的派生优化器。
# AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')

model.compile(
    loss='sparse_categorical_crossentropy',
    # optimizer=Adam(1e-5),  # 用足够小的学习率
    optimizer=Adam(learning_rate=1e-5), # AdamLR(learning_rate=1e-4, lr_schedule={1000: 1, 2000: 0.1}),
    metrics=['accuracy'],
)

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model.weights')
        # test_acc = evaluate()
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )


evaluator = Evaluator()
model.fit_generator(
    train_generator.forfit(),
    steps_per_epoch=len(train_generator),
    epochs=10,
    callbacks=[evaluator]
)

# model.load_weights('best_model.weights')
# print(u'final test acc: %05f\n' % (evaluate(test_generator)))
