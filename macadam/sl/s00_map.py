# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/5/9 21:55
# @author  : Mo
# @function: graph mapping of sequence-labeling(include ner, pos, tag)


from macadam.sl.s02_bilstm_crf import BiLstmCRFGraph, BiGruCRFGraph
from macadam.sl.s06_lattice_lstm_batch import LatticeLSTMBatchgraph
from macadam.sl.s05_bilstm_lan import BiLstmLANGraph
from macadam.sl.s03_cnn_lstm import CnnLstmGraph
from macadam.sl.s04_dgcnn import DGCNNGraph
from macadam.sl.s01_crf import CRFGraph


graph_map = {"LATTICE-LSTM-BATCH": LatticeLSTMBatchgraph,
             "BI-LSTM-CRF": BiLstmCRFGraph,
             "BI-LSTM-LAN": BiLstmLANGraph,
             "BI-GRU-CRF": BiGruCRFGraph,
             "CNN-LSTM": CnnLstmGraph,
             "DGCNN": DGCNNGraph,
             "CRF": CRFGraph,
             }
