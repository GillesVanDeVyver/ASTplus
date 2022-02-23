# -*- coding: utf-8 -*-
# @Time    : 6/24/21 12:50 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : demo.py

import os
import torch
from models import ast_models
# download pretrained model in this directory
os.environ['TORCH_HOME'] = '../pretrained_models'
# assume each input spectrogram has 100 time frames
input_tdim = 100
# assume the task has 527 classes
label_dim = 527
# create a pseudo input: a batch of 10 spectrogram, each with 100 time frames and 128 frequency bins
test_input = torch.rand([10, input_tdim, 128])
# create an AST model
ast_mdl = ast_models.ASTModel(label_dim=label_dim, input_tdim=input_tdim, imagenet_pretrain=False,
                              audioset_pretrain=False)


trainables = [p for p in ast_mdl.parameters() if p.requires_grad]
print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in ast_mdl.parameters()) / 1e6))
print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))


test_output = ast_mdl(test_input)
# output should be in shape [10, 527], i.e., 10 samples, each with prediction of 527 classes.
print(test_output.shape)
print(test_input.shape)