#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com), Xiangtai(lxtpku@pku.edu.cn)
# Select Seg Model for semantic segmentation.


import torch


from models.seg.nets.annn import asymmetric_non_local_network
from models.seg.loss.seg_modules import FSCELoss, FSOhemCELoss, FSAuxCELoss, FSAuxEncCELoss, FSAuxOhemCELoss
from utils.tools.logger import Logger as Log

SEG_MODEL_DICT = {
    'annn': asymmetric_non_local_network
}

SEG_LOSS_DICT = {
    'fs_ce_loss': FSCELoss,
    'fs_ohemce_loss': FSOhemCELoss,
    'fs_auxce_loss':FSAuxCELoss,
    'fs_auxencce_loss': FSAuxEncCELoss,
    'fs_auxohemce_loss': FSAuxOhemCELoss
}

class ModelManager(object):

    def __init__(self, configer):
        self.configer = configer

    def semantic_segmentor(self):
        model_name = self.configer.get('network', 'model_name')

        if model_name not in SEG_MODEL_DICT:
            Log.error('Model: {} not valid!'.format(model_name))
            exit(1)

        model = SEG_MODEL_DICT[model_name](self.configer)

        return model

    def get_seg_loss(self, loss_type=None):
        key = self.configer.get('loss', 'loss_type') if loss_type is None else loss_type
        if key not in SEG_LOSS_DICT:
            Log.error('Loss: {} not valid!'.format(key))
            exit(1)

        loss = SEG_LOSS_DICT[key](self.configer)
        if self.configer.get('network', 'loss_balance') and len(range(torch.cuda.device_count())) > 1:
            from extensions.tools.parallel.data_parallel import DataParallelCriterion
            loss = DataParallelCriterion(loss)

        return loss
