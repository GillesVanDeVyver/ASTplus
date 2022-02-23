# -*- coding: utf-8 -*-
# @Time    : 6/10/21 11:00 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : traintest.py

import sys
import os
import datetime

import pandas as pd

import torch

sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
import time
import torch
from torch import nn
import numpy as np
import pickle
from torch.cuda.amp import autocast,GradScaler
import ASTAE_model


"""
Some parts are taken from AST: https://github.com/YuanGongND/ast
    especially (initial) hyper parameters
    parts copied from AST are annotated with #AST
"""


def train(audio_model, input_base_directory, batch_size, lr,lr_patience=10,n_epochs=1,verbose=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

#    audio_model = audio_model.to(device)

    # Set up the optimizer #AST
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    optimizer = torch.optim.Adam(trainables, lr, weight_decay=5e-7, betas=(0.95, 0.999))

    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=lr_patience,
                                                           #verbose=verbose) #AST
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2,3,4,5], gamma=0.5, last_epoch=-1) #AST
    loss_fn = torch.nn.MSELoss()

    # for amp # AST
    scaler = GradScaler()

    batch_size=1


    global_step, epoch = 0, 0
    result = np.zeros([n_epochs, 10])
    audio_model.train()

    dataframes_base_directory = "../../dev_data_dataframes/"
    #for machine in os.listdir(dataframes_base_directory):
    machine = "debug_sample"

    dataframe_dir = input_base_directory + machine+"/"
    print(dataframe_dir)
    temp = dataframe_dir+"train/dataframe.pt"
    X_train = torch.load(temp)
    #X_train.to(device, non_blocking=True)

    nb_batches = round(len(X_train)/batch_size)

    while epoch < n_epochs + 1:
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))
        pos = 0
        for i in range(nb_batches):
            print(i)
            temp = len(X_train)
            if (pos + batch_size>len(X_train)):
                X_batch = X_train[pos:]
            else:
                X_batch = X_train[pos:pos + batch_size]
            #X_batch=X_batch.to(device, non_blocking=True)
            #labels = labels.to(device, non_blocking=True)
#                audio_model = audio_model.to(device)
            with autocast():
                audio_output,lin_proj_output = audio_model(X_batch)
                loss = loss_fn(audio_output, lin_proj_output)

            optimizer.zero_grad()
            if device=="cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            loss.backward()

            global_step += 1

        scheduler.step()
        epoch += 1



#TODO
"""
- Add decoder block
- Run on server instead to avoid out of memory problems
- adjust hyper parameters
- put hyper parameters in yaml file

extras not taken over from AST
-> warmup
-> save intermediate progress
"""


audio_model = ASTAE_model.ASTAEModel()
input_base_directory = "../../dev_data_dataframes/"
lr = 0.000005
batch_size=1
train(audio_model, input_base_directory, batch_size, lr,lr_patience=10,n_epochs=2,verbose=True)


"""
input_base_directory = "../../dev_data_dataframes/"


dataframe_dir = input_base_directory + "pump"+"/"

temp = dataframe_dir+"train/dataframe.pt"

temp_test = X_train = torch.load(temp)

print(len(temp_test))
"""