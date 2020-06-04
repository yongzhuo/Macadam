# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/5/8 21:05
# @author  : Mo
# @function: Capsule


from macadam.base.layers import Capsule_bojone
from macadam.base.graph import graph
from macadam import keras, K, L, M


class CapsuleGraph(graph):
    def __init__(self, hyper_parameters):
        self.routings = hyper_parameters.get("graph", {}).get("routings", 5)
        self.dim_capsule = hyper_parameters.get("graph", {}).get("dim_capsule", 16)
        self.num_capsule = hyper_parameters.get("graph", {}).get("num_capsule", 16)
        self.dropout_spatial = hyper_parameters.get("graph", {}).get("dropout_spatial", 0.2)
        super().__init__(hyper_parameters)

    def build_model(self, inputs, outputs):
        outputs_spati = L.SpatialDropout1D(self.dropout_spatial)(outputs)

        conv_pools = []
        for filter in self.filters_size:
            x = L.Conv1D(filters=self.filters_num,
                         kernel_size=filter,
                         padding="valid",
                         kernel_initializer="normal",
                         activation="relu",
                         )(outputs_spati)
            capsule = Capsule_bojone(num_capsule=self.num_capsule,
                                     dim_capsule=self.dim_capsule,
                                     routings=self.routings,
                                     kernel_size=(filter, 1),
                                     share_weights=True)(x)
            conv_pools.append(capsule)
        capsule = L.Concatenate(axis=-1)(conv_pools)
        x = L.Flatten()(capsule)
        x = L.Dropout(self.dropout)(x)
        # dense-mid
        x = L.Dense(units=min(max(self.label, 64), self.embed_size), activation=self.activate_mid)(x)
        x = L.Dropout(self.dropout)(x)
        # dense-end, 最后一层, dense到label
        self.outputs = L.Dense(units=self.label, activation=self.activate_end)(x)
        self.model = M.Model(inputs=inputs, outputs=self.outputs)
        self.model.summary(132)

