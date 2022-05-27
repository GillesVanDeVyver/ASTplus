import os
import numpy as np
import yaml
import pandas as pd
import common
import csv
import sys
from sklearn import metrics
try:
    from sklearn.externals import joblib
except:
    import joblib
import torch
import warmup_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast,GradScaler
import datetime
import matplotlib.pyplot as plt
import adaptedAST

#device = 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eps = 0.1



def custom_plot(epochs, loss,figname,debug):
    plt.plot(epochs, loss)
    if debug:
        plt.savefig("debug_results/"+figname+".png")
    else:
        plt.savefig("results/"+figname+".png")

def calc_AUC(X,X_indices,labels,loss_fn,source,log=True,tb=None,epoch=None,debug=False,max_fpr=0.1,device=device):
    if log and tb==None:
        raise Exception("no tensorboard to log to given")
    mse = []
    for i in range(len(X)):
        sample = X[i]
        sample_index=X_indices[i]

        with autocast():
            sample = torch.unsqueeze(sample, dim=0)
            #if device == "cuda":
            sample=sample.to(device)
            sample_index=sample_index.to(device)
            sample_output = audio_model(sample.detach())
            sample_output = sample_output.detach()
            sample_output=torch.flatten(sample_output,0)
            sample_loss = loss_fn(sample_output, sample_index)
        mse.append(sample_loss.item())
    auc = metrics.roc_auc_score(labels, mse)
    pauc= metrics.roc_auc_score(labels, mse,max_fpr=max_fpr)
    if log:
        label = 'target'
        if source:
            label = 'source'
        tb.add_scalar('AUC_scores/'+label, auc, epoch)
        tb.add_scalar('pAUC_scores/'+label, pauc,epoch)
    return mse,auc,pauc

def generate_roc_curve(y,labels,title):
    fpr_source, tpr_source, thresholds_source = metrics.roc_curve(labels, y)
    ROC_location_source="results/"+title+".png"
    common.generate_ROC_curve(fpr_source, tpr_source, ROC_location_source)


def convert_to_one_hot(X,soft=True,nb_sections=3,eps=0.1):
    if soft:
        one_hot_labels = np.full((len(X), nb_sections),eps,dtype=float)
        for i in range(len(X)):
            section_index=X[i]
            one_hot_labels[i,section_index] = 1-(nb_sections-1)*eps
    else:
        raise 'hard labels not implemented yet'
    return one_hot_labels


def train(version,model_title,audio_model, input_base_directory, batch_size, lr,machine,
          lr_patience=10,n_epochs=1,verbose=True,debug=False,warmup=True,shuffle=True):

    print('running on ' + str(device))
    torch.set_grad_enabled(True)
    audio_model = audio_model.to(device) #!!!

    # Set up the optimizer #AST
    trainable_params = [p for p in audio_model.parameters() if p.requires_grad]

    print('Number of trainable parameters is : {:.3f} million'.format(sum(p.numel() for p in trainable_params) / 1e6))

    optimizer = torch.optim.Adam(
        [{'params': trainable_params, 'lr': lr}], betas=(0.95, 0.999)
    ) #AST

    if warmup:
        scheduler = warmup_scheduler.WarmupLR(optimizer,warmup_steps=10)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2,3,4,5], gamma=0.5, last_epoch=-1)

    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=lr_patience,
            #threshold=0.01, threshold_mode='abs',verbose=verbose) #AST
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [25,60,100,150], gamma=0.5, last_epoch=-1) #AST
    #if warmup:
    #    scheduler = warmup_scheduler.WarmupLR(optimizer,warmup_steps=10)
    #else:
    #    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [25,60,100,150], gamma=0.5, last_epoch=-1)

    loss_fn = torch.nn.MSELoss()

    # for amp # AST
    scaler = GradScaler()

    global_step, epoch = 0, 1
    audio_model.train()

    dataframe_dir = input_base_directory + machine+"/"
    train_location = dataframe_dir+"train/dataframe.pt"
    train_indices_location = dataframe_dir+"train/index_labels.pt"
    validation_source_location = dataframe_dir+"source_test/dataframe.pt"
    validation_source_indices_location = dataframe_dir+"source_test/index_labels.pt"
    validation_source_labels_location = dataframe_dir+"source_test/labels.pt"
    validation_target_location = dataframe_dir+"target_test/dataframe.pt"
    validation_target_indices_location = dataframe_dir+"target_test/index_labels.pt"
    validation_target_labels_location = dataframe_dir+"target_test/labels.pt"


    X_train = torch.load(train_location)
    X_train_indices=torch.load(train_indices_location)
    temp = max(X_train_indices)
    X_validation_source = torch.load(validation_source_location)
    X_validation_source_indices = torch.load(validation_source_indices_location)
    X_validation_source_labels = torch.load(validation_source_labels_location)
    X_validation_target = torch.load(validation_target_location)
    X_validation_target_indices = torch.load(validation_target_indices_location)
    X_validation_target_labels = torch.load(validation_target_labels_location)



    if debug:
        X_validation_source = X_validation_source[:6]
        X_validation_source_labels = X_validation_source_labels[:6]
        X_validation_source_indices= X_validation_source_indices[:6]
        X_validation_target = X_validation_target[:6]
        X_validation_target_labels = X_validation_target_labels[:6]
        X_validation_target_indices= X_validation_target_indices[:6]

    X_train.to(device, non_blocking=True) #!!
    X_train_indices=torch.as_tensor(X_train_indices)
    X_train_indices.to(device, non_blocking=True)
    X_validation_source.to(device, non_blocking=True)
    X_validation_source_indices=torch.as_tensor(X_validation_source_indices)
    X_validation_source_indices.to(device, non_blocking=True)
    X_validation_target.to(device, non_blocking=True)
    X_validation_target_indices=torch.as_tensor(X_validation_source_indices)
    X_validation_target_indices.to(device, non_blocking=True)

    nb_batches = round(len(X_train)/batch_size)
    title = "DCASE21_"+model_title+"_"+machine+"_"+version
    if debug:
        log_folder = "runs/debug/"+machine+"/"
    else:
        log_folder = "runs/"+machine+"/"

    torch.cuda.empty_cache() #!!!!
    tb = SummaryWriter(log_folder+title)
    #sample_input = X_train[0:batch_size]
    #sample_input=sample_input.to(device) #!
    #tb.add_graph(audio_model,sample_input)
    audio_model = torch.nn.DataParallel(audio_model)
    train_loss_vals=  []
    epoch = 1
    if debug:
        n_epochs = 5
    while epoch < n_epochs:
        if shuffle:
            perm = torch.randperm(X_train.size()[0])
            X_train=X_train[perm]
            X_train_indices=X_train_indices[perm]
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))
        pos = 0
        epoch_loss= []
        if debug:
            nb_batches = 2
        audio_model.train()
        for i in range(nb_batches):
            if (i%100 == 0):
                print(i)
            if (pos + batch_size>len(X_train)):
                X_batch = X_train[pos:]
                X_batch_indices=X_train_indices[pos:]
            else:
                X_batch = X_train[pos:pos + batch_size]
                X_batch_indices=X_train_indices[pos:pos + batch_size]
            X_batch=X_batch.to(device, non_blocking=True) #!!!
#            labels = labels.to(device, non_blocking=True)
#                audio_model = audio_model.to(device)
            with autocast():
                audio_output = audio_model(X_batch)
                X_batch_one_hot_soft = convert_to_one_hot(X_batch_indices)
                X_batch_one_hot_soft=torch.as_tensor(X_batch_one_hot_soft)
                X_batch_one_hot_soft=X_batch_one_hot_soft.to(device)
                X_batch_one_hot_soft=X_batch_one_hot_soft.to(torch.float32)
                audio_output=audio_output.to(device)
                loss = loss_fn(audio_output, X_batch_one_hot_soft)


            optimizer.zero_grad()
            if device=="cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                #loss=loss.to(torch.float)
                loss.backward()
                optimizer.step()
            epoch_loss.append(loss.item())

            global_step += 1
        avg_epoch_loss=sum(epoch_loss)/len(epoch_loss)
        train_loss_vals.append(avg_epoch_loss)
        tb.add_scalar('Loss/train', avg_epoch_loss, epoch)

        scheduler.step(avg_epoch_loss)


        audio_model.eval() # log validation accuracy during training (and without interfering with training)

        X_validation_source_indices_one_hot=torch.as_tensor(convert_to_one_hot(X_validation_source_indices))
        mse_source,auc_source,pauc_source = calc_AUC(X_validation_source,X_validation_source_indices_one_hot,
                                                     X_validation_source_labels,loss_fn,True,log=True,tb=tb,
                                                     epoch=epoch,debug=debug,
                                                     )
        X_validation_target_indices_one_hot=torch.as_tensor(convert_to_one_hot(X_validation_target_indices))
        mse_target,auc_target,pauc_target= calc_AUC(X_validation_target,X_validation_target_indices_one_hot,
                                                    X_validation_target_labels,loss_fn,False,log=True,tb=tb,
                                                    epoch=epoch,debug=debug,
                                                    )
        #if epoch%50==0 and not debug:
        #    torch.save(audio_model.state_dict(), "trained_models/intermediate_results/"+title+"_intermediate_"+str(epoch)+".pt")

        epoch += 1




    #figname = "DCASE21_"+machine+"_"+version+"_train"
    #custom_plot(np.linspace(1, n_epochs, n_epochs).astype(int), train_loss_vals, figname,debug)
    #for n_iter in range(100):
     #   tb.add_scalar('Loss/train', np.random.random(), n_iter)
    if not debug:
        torch.save(audio_model.state_dict(), "trained_models/"+title+".pt")

    #generate_roc_curve(mse_source,X_validation_source_labels,title+"_source")
    #generate_roc_curve(mse_target,X_validation_target_labels,title+"_target")


    tb.close()











server = False
debug=False
version = "no version set"

nb_enc_layers = 3
depth_trainable = 1
batch_size=8
#batch_size=32  #128 makes gpu go out of memory 16 on own device
n_epochs=400
lr = 0.000005

warmup = True


versions = ['8init']

for i in range(len(versions)):
    version = versions[i]

    for machine in ["gearbox"]:

        audio_model = adaptedAST.ASTModel(imagenet_pretrain=True, audioset_pretrain=True, verbose=True,label_dim_new=3,
                                          depth_encoder=nb_enc_layers,depth_trainable=depth_trainable,device=device)

        model_title = "ASTOD"

        if server:
            input_base_directory = "../../dev_data_dataframes_server/"
        else:
            input_base_directory = "../../dev_data_dataframes/"

        train(version,model_title,audio_model, input_base_directory, batch_size, lr,machine=machine,
          lr_patience=5,n_epochs=n_epochs,verbose=True,debug=debug,warmup=warmup,shuffle=True)

        print(machine + " done")











