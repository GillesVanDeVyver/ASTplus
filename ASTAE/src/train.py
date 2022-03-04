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
import pureAttentionAE
import attention_linear
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
import common


"""
Some parts are taken from AST: https://github.com/YuanGongND/ast
    especially (initial) hyper parameters
    parts copied from AST are annotated with #AST
"""

def custom_plot(epochs, loss,figname,debug):
    plt.plot(epochs, loss)
    if debug:
        plt.savefig("debug_results/"+figname+".png")
    else:
        plt.savefig("results/"+figname+".png")

def calc_AUC(X,labels,loss_fn,source,log=True,tb=None,epoch=None,debug=False,max_fpr=0.1,device='cuda'):
    if log and tb==None:
        raise Exception("no tensorboard to log to given")
    if log and tb==None:
        raise Exception("no epoch given for logging")
    mse = []
    for sample in X:
        with autocast():
            sample = torch.unsqueeze(sample, dim=0)
            sample=sample.to(device)
            sample_output,sample_lin_proj_output = audio_model(sample.detach())
            sample_output = sample_output.detach()
            sample_lin_proj_output = sample_lin_proj_output.detach()
            sample_loss = loss_fn(sample_output, sample_lin_proj_output)
        mse.append(sample_loss.item())
    auc = metrics.roc_auc_score(labels, mse)
    pauc= metrics.roc_auc_score(labels, mse,max_fpr=max_fpr)
    if log:
        label = 'target'
        if source:
            label = 'source'
        tb.add_scalar('AUC_scores/'+label, auc, epoch)
        tb.add_scalar('pAUC_scores/'+label, pauc, epoch)
    return mse,auc,pauc

def generate_roc_curve(y,labels,title):
    fpr_source, tpr_source, thresholds_source = metrics.roc_curve(labels, y)
    ROC_location_source="results/"+title+".png"
    common.generate_ROC_curve(fpr_source, tpr_source, ROC_location_source)


def train(version,model_title,audio_model, input_base_directory, batch_size, lr,lr_patience=10,n_epochs=1,verbose=True,debug=False):





    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    audio_model = audio_model.to(device)

    # Set up the optimizer #AST
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    optimizer = torch.optim.Adam(trainables, lr, weight_decay=5e-7, betas=(0.95, 0.999)) #AST

    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=lr_patience,
                                                           #verbose=verbose) #AST
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2,3,4,5], gamma=0.5, last_epoch=-1) #AST
    loss_fn = torch.nn.MSELoss()

    # for amp # AST
    scaler = GradScaler()



    global_step, epoch = 0, 1
    result = np.zeros([n_epochs, 10])
    audio_model.train()

    #dataframes_base_directory = "../../dev_data_dataframes/"
    #for machine in os.listdir(dataframes_base_directory):

    for machine in ["fan","gearbox","pump","slider","ToyCar","ToyTrain","valve"]:
    #for machine in ["fan"]:



        dataframe_dir = input_base_directory + machine+"/"
        train_location = dataframe_dir+"train/dataframe.pt" #labels unecessary: all normal
        validation_source_location = dataframe_dir+"source_test/dataframe.pt"
        validation_source_labels_location = dataframe_dir+"source_test/labels.pt"
        validation_target_location = dataframe_dir+"target_test/dataframe.pt"
        validation_target_labels_location = dataframe_dir+"target_test/labels.pt"

        X_train = torch.load(train_location)
        X_validation_source = torch.load(validation_source_location)
        X_validation_source_labels = torch.load(validation_source_labels_location)
        X_validation_target = torch.load(validation_target_location)
        X_validation_target_labels = torch.load(validation_target_labels_location)
        if debug:
            X_validation_source = X_validation_source[:6]
            X_validation_source_labels = X_validation_source_labels[:6]
            X_validation_target = X_validation_target[:6]
            X_validation_target_labels = X_validation_target_labels[:6]
        X_train.to(device, non_blocking=True)
        X_validation_source.to(device, non_blocking=True)
        X_validation_target.to(device, non_blocking=True)


        nb_batches = round(len(X_train)/batch_size)

        title = "DCASE21_"+model_title+"_"+machine+"_"+version

        if debug:
            log_folder = "runs/debug/"+machine+"/"
        else:
            log_folder = "runs/"+machine+"/"




        tb = SummaryWriter(log_folder+title)
        sample_input = X_train[0:batch_size]
        sample_input=sample_input.to(device)
        tb.add_graph(audio_model,sample_input)

        train_loss_vals=  []
        epoch = 0
        while epoch < n_epochs + 1:
            print('---------------')
            print(datetime.datetime.now())
            print("current #epochs=%s, #steps=%s" % (epoch, global_step))
            pos = 0
            epoch_loss= []
            if debug:
                nb_batches = 2


            audio_model .train()
            for i in range(nb_batches):
                if (i%100 == 0):
                    print(i)
                if (pos + batch_size>len(X_train)):
                    X_batch = X_train[pos:]
                else:
                    X_batch = X_train[pos:pos + batch_size]
                X_batch=X_batch.to(device, non_blocking=True)
    #            labels = labels.to(device, non_blocking=True)
    #                audio_model = audio_model.to(device)
                with autocast():
                    audio_output,lin_proj_output = audio_model(X_batch)
                    loss = loss_fn(audio_output, lin_proj_output)


                optimizer.zero_grad()
                if device=="cuda":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                epoch_loss.append(loss.item())

                global_step += 1
            avg_epoch_loss=sum(epoch_loss)/len(epoch_loss)
            train_loss_vals.append(avg_epoch_loss)
            tb.add_scalar('Loss/train', avg_epoch_loss, epoch)

            scheduler.step()


            audio_model.eval() # log validation accuracy during training (and without interfering with training)

            mse_source,auc_source,pauc_source = calc_AUC(X_validation_source,X_validation_source_labels,loss_fn,True,log=True,tb=tb,epoch=epoch,debug=debug)
            mse_target,auc_target,pauc_target= calc_AUC(X_validation_target,X_validation_target_labels,loss_fn,False,log=True,tb=tb,epoch=epoch,debug=debug)

            epoch += 1

        #figname = "DCASE21_"+machine+"_"+version+"_train"
        #custom_plot(np.linspace(1, n_epochs, n_epochs).astype(int), train_loss_vals, figname,debug)
        #for n_iter in range(100):
         #   tb.add_scalar('Loss/train', np.random.random(), n_iter)
        if not debug:
            torch.save(audio_model.state_dict(), "trained_models/"+title+".pt")

        generate_roc_curve(mse_source,X_validation_source_labels,title)
        generate_roc_curve(mse_target,X_validation_target_labels,title)


        tb.close()

        print(machine + " done")


#TODO
"""
- adjust hyper parameters?
- put hyper parameters in yaml file

extras not taken over from AST
-> warmup
-> save intermediate progress
"""



server = True

audio_model = attention_linear.attention_linear_model(depth_encoder=1, trainable_encoder=False,avg=True)
model_title = "attention_linear"


if server:
    input_base_directory = "../../dev_data_dataframes_server/"
else:
    input_base_directory = "../../dev_data_dataframes/"



lr = 0.000005
batch_size=12 # AST
version = "4.0"
train(version,model_title,audio_model, input_base_directory, batch_size, lr,lr_patience=10,n_epochs=200,verbose=True,
      debug=True)


"""
input_base_directory = "../../dev_data_dataframes/"


dataframe_dir = input_base_directory + "pump"+"/"

temp = dataframe_dir+"train/dataframe.pt"

temp_test = X_train = torch.load(temp)

print(len(temp_test))
"""

#plot_loss([1, 2, 3, 4, 5], [100, 90, 60, 30, 10],figname="test")
