import sys


import os

import PIL
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

import cv2
from PIL import Image
from torchvision import transforms

from ast_based_novelty_visualization import adast

from spectrogram_plots import convert_to_spectrogram_and_save
from sklearn import preprocessing
import gc


transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def get_attention_map(input_location,img, model,get_mask=False):

    att_mat = model.get_ast_embedding_single_file(input_location)


    att_mat = torch.stack(att_mat).squeeze(1)

    att_mat=att_mat.to('cpu')


    #print(att_mat.detach().numpy().shape)

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)


    #print(att_mat.detach().numpy().shape)
    aug_att_mat=att_mat

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    #residual_att = torch.eye(att_mat.size(1))
    #aug_att_mat = att_mat + residual_att
    #temp = aug_att_mat.sum(dim=-1)
    #temp2 = aug_att_mat.sum(dim=-1).unsqueeze(-1)

    #aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)



    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
    #for n in range(1,3):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    v = joint_attentions[-1]
    tempv=v.detach().numpy()

    mask = (v[0,2:].detach().numpy()+v[1,2:].detach().numpy())/2



    mask = mask.reshape(12, 101)
    temp = mask[:2,:]
    mask[:2,:]=mask[:2,:]/10 # adjust for the fact that high up in the frequency spectrum the values are lower
    mask = mask[:,:-2] # adjust for padding
    #mask = (v[0, 2:].reshape(12, 101).detach().numpy()+v[1, 2:].reshape(12, 101).detach().numpy())/2
    #mask = (v[2:, 0].reshape(12, 101).detach().numpy()+v[2:, 1].reshape(12, 101).detach().numpy())/2

    mask = mask[2:,:-2]
    #mask=np.flip(mask,0)

    #img2 = img.convert('RGBA')
    #arr = np.array(img2)

    if get_mask:
        result = cv2.resize(mask / mask.max(), img.size)
    else:
        mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
        #mask_avg = np.average(mask)
        #mask=mask-mask_avg
        mask_avg = np.average(mask)
        mask=mask/mask_avg
        #max_mask = np.amax(mask)
        #mask=mask/max_mask
        #mask = mask-mask_avg
        result = (mask * img).astype("uint8")
        #result = mask*arr
        for row in result:
            for culumn in row:
                for i in range(3):

                    if culumn[i]>=255:
                        culumn[i]=255

                culumn[3]=255

    return result


def plot_attention_map(original_img, att_map):
    plt.figure()
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    ax1.set_title('Original')
    ax2.set_title('Attention Map Last Layer')
    _ = ax1.imshow(original_img)
    _ = ax2.imshow(att_map)



def plot_attention_map_and_safe(original_img, att_map,output_location):
    plt.figure()
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    ax1.set_title('Original')
    ax2.set_title('Attention Map Last Layer')
    _ = ax1.imshow(original_img)
    _ = ax2.imshow(att_map)
    plt.savefig(os.path.join(output_location))
    fig.clf()
    plt.clf()
    plt.close("all")
    del att_map
    del original_img
    del fig
    gc.collect()


"""
# generate spectrograms of some samples of the data
base_directory="../../dev_data/"
output_base_directory="../results/spectrograms/"
for machine in os.listdir(base_directory):
    for domain in os.listdir(base_directory+"/"+machine):
        input_directory = base_directory + machine + "/" + domain
        output_directory = output_base_directory + machine+'/'+domain
        count = 0
        for filename in os.listdir(input_directory):
            if filename.endswith(".wav") and count % 100 == 0:
                file_location = os.path.join(input_directory, filename)
                sample_name = os.path.splitext(file_location[len(input_directory):])[0]
                output_location = output_directory + sample_name + ".png"
                convert_to_spectrogram_and_save(file_location, output_location)
            count+=1

        print(machine+" "+domain+" done")
"""


# generate attation maps of the samples

adast_mdl = adast()

start_interval= 65 +40+ 30 +40 + 30 +40 + 30 +40 + 35 +40 + 30 +40 + 25
end_interval = start_interval+4

base_directory_spectrograms="./results/spectrograms/"
base_directory="../../dev_data/"
output_base_directory="./results/attention_maps12_hack2/"
skipcount=0
#for machine in os.listdir(base_directory_spectrograms):
for machine in ['ToyTrain','gearbox','fan']:
    for domain in ['train']:
        skipcount=0

        torch.cuda.empty_cache()
        print("cache cleared")
    #for domain in os.listdir(base_directory_spectrograms+"/"+machine):
        input_directory_specrograms = base_directory_spectrograms + machine + "/" + domain
        input_directory = base_directory+ machine + "/" + domain
        output_directory = output_base_directory + machine+'/'+domain
        for filename in os.listdir(input_directory_specrograms):
            if filename.endswith(".png") and skipcount>0 and skipcount<5 :
                print(skipcount)
                torch.cuda.empty_cache()
                print("cache cleared")

                wav_file_location = os.path.join(input_directory, filename[:-3] + "wav")
                png_file_location = os.path.join(input_directory_specrograms, filename)


                sample_name = os.path.splitext(wav_file_location[len(input_directory):])[0]
                output_location = output_directory + sample_name + "_attention_map.png"

                img=Image.open(png_file_location)
                result = get_attention_map(wav_file_location,img,adast_mdl)
                plot_attention_map_and_safe(img, result,output_location)
                gc.collect()
            skipcount+=1

        print(machine+" "+domain+" done")


