# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/5/8 16:38
# @author  : Mo
# @function: graph mapping of text-classification


from macadam.tc.t01_finetune import FineTuneGraph
from macadam.tc.t02_fasttext import FastTextGraph
from macadam.tc.t03_textcnn import TextCNNGraph
from macadam.tc.t04_charcnn import CharCNNGraph
from macadam.tc.t05_birnn import BiRNNGraph
from macadam.tc.t06_rcnn import RCNNGraph
from macadam.tc.t07_dcnn import DCNNGraph
from macadam.tc.t08_crnn import CRNNGraph
from macadam.tc.t09_deepmoji import DeepMojiGraph
from macadam.tc.t10_attention import SelfAttentionGraph
from macadam.tc.t11_capsule import CapsuleGraph


graph_map = {"SELFATTENTION": SelfAttentionGraph,
             "DEEPMOJI": DeepMojiGraph,
             "CAPSULE": CapsuleGraph,
             "FINETUNE": FineTuneGraph,
             "FASTTEXT": FastTextGraph,
             "TEXTCNN": TextCNNGraph,
             "ChARCNN": CharCNNGraph,
             "BIRNN": BiRNNGraph,
             "RCNN": RCNNGraph,
             "DCNN": DCNNGraph,
             "CRNN": CRNNGraph,
             }

