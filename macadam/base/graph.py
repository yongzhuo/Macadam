# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2020/4/30 20:51
# @author   :Mo
# @function :graph of base


# from macadam.base.preprocess import ListGenerator, FileGenerator, ListPrerocessXY, Tokenizer4Macadam
# from macadam.conf.path_config import path_model, path_model_dir, path_fineture, path_model_info
from macadam.base.utils import load_json, save_json, txt_read, txt_write
from macadam.conf.constant_params import CLS, SEP, PAD, UNK, MASK
from macadam.conf.constant_params import SL, TC, RE
from macadam.conf.constant_params import Config
from macadam.conf.logger_config import logger
from macadam import keras, K, L, M, O, C
from macadam import __version__
## optimizers
# from bert4keras.optimizers import extend_with_gradient_accumulation
# from bert4keras.optimizers import extend_with_piecewise_linear_lr
# from bert4keras.optimizers import extend_with_layer_adaptation
# from bert4keras.optimizers import extend_with_lazy_optimization
# from bert4keras.optimizers import extend_with_weight_decay
# from bert4keras.optimizers import extend_with_lookahead
# from bert4keras.optimizers import AdaFactor
# from bert4keras.optimizers import Adam
## common
import numpy as np
import os


class graph(Config):
    def __init__(self, hyper_parameters):
        self.model = None
        self.trans = None
        super().__init__(hyper_parameters)

    def build_model(self, inputs, outputs):
        self.model = None

    def create_compile(self):
        """
          构建优化器、损失函数和评价函数
        :return: 
        """
        # optimizer_ = Adam if self.optimizer.upper() == "ADAM" else AdaFactor
        # if "gradient_accumulation" in self.optimizer_extend:
        #     optimizer_ = extend_with_gradient_accumulation(optimizer_)
        # if "piecewise_linear_lr" in self.optimizer_extend:
        #     optimizer_ = extend_with_piecewise_linear_lr(optimizer_)
        # if "layer_adaptation" in self.optimizer_extend:
        #     optimizer_ = extend_with_layer_adaptation(optimizer_)
        # if "lazy_optimization" in self.optimizer_extend:
        #     optimizer_ = extend_with_lazy_optimization(optimizer_)
        # if "weight_decay" in self.optimizer_extend:
        #     optimizer_ = extend_with_weight_decay(optimizer_)
        # if "lookahead" in self.optimizer_extend:
        #     optimizer_ = extend_with_lookahead(optimizer_)

        self.model.compile(optimizer=O.Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, decay=0.0),
                           loss=self.loss,
                           metrics=[self.metrics])

    def callback(self, monitor="val_loss"):
        """
          评价函数、早停
        :return: 
        """
        cb_em = [C.TensorBoard(log_dir=os.path.join(self.path_model_dir, "logs"), batch_size=self.batch_size, update_freq='batch'),
                 C.EarlyStopping(monitor=monitor, mode="auto", min_delta=1e-9, patience=self.early_stop),
                 C.ModelCheckpoint(monitor=monitor, mode="auto", filepath=os.path.join(self.path_model_dir, "macadam.h5"),
                                   verbose=1, save_best_only=True, save_weights_only=False)]
        return cb_em

    def fit(self, preprocess_xy, generator_xy, train_data, dev_data=None, rate=1.0):
        """
        fit_generator, 迭代器训练
        Args:
            preprocess_xy: class, alreadly init
            generator_xy: class, not init
            train_data: List or path
            dev_data: List or path
        Returns:
            None
        """
        # 保存所有需要的参数, 包括macadam版本/模型标签字典
        self.hyper_parameters["train"]["is_training"] = False  # 预测时候这些设为False
        self.hyper_parameters["train"]["trainable"] = False
        self.hyper_parameters["graph"]["dropout"] = 0.0
        # 保存的超参数, info
        model_info = {"__class__": self.__class__.__name__,
                      "__version__": __version__,
                      "hyper_parameters": self.hyper_parameters,
                      "label": {"l2i": preprocess_xy.l2i,
                                "i2l": preprocess_xy.i2l},
                      "vocab": {"token2idx": preprocess_xy.embedding.token2idx
                                },
                      }
        save_json(lines=model_info, path=os.path.join(self.path_model_dir, "macadam.info"))
        # 训练数据集的数据条数
        len_train = preprocess_xy.analysis_len_data(train_data)
        len_train = int(len_train * rate)
        lg_train = generator_xy(train_data, preprocess_xy,
                                batch_size=self.batch_size, len_data=len_train)
        lg_dev = None
        # monitor是早停和保存模型的依据, "loss", "acc", "val_loss", "val_acc"等
        monitor = "val_loss"
        if dev_data:
            len_dev = preprocess_xy.analysis_len_data(dev_data)
            len_dev= int(len_dev * rate)
            lg_dev = generator_xy(dev_data, preprocess_xy,
                                  batch_size=self.batch_size, len_data=len_dev)
        else:
            monitor = "loss"
        # 训练模型
        self.model.fit_generator(generator=lg_train.forfit(),
                                 steps_per_epoch=lg_train.__len__(),
                                 callbacks=self.callback(monitor),
                                 epochs=self.epochs,
                                 validation_data=lg_dev.forfit() if lg_dev else None,
                                 validation_steps=lg_dev.__len__() if lg_dev else None)
        # 保存crf状态转移矩阵
        if self.use_crf and self.trans:
            model_info = {"__class__": self.__class__.__name__,
                          "__version__": __version__,
                          "hyper_parameters": self.hyper_parameters,
                          "label": {"l2i": preprocess_xy.l2i,
                                    "i2l": preprocess_xy.i2l},
                          SL: {"trans": self.trans},
                          "vocab": {"token2idx": preprocess_xy.embedding.token2idx},
                          }
            save_json(lines=model_info, path=os.path.join(self.path_model_dir, "macadam.info"))

    def add_metrics(self, metrics):
        self.metrics = metrics

    def add_loss(self, loss):
        self.loss = loss

