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

from ast_based_novelty_visualization import adast

from spectrogram_plots import convert_to_spectrogram_and_save
from sklearn import preprocessing
import gc










def save_spectrogram(original_img, att_map,output_location):
    plt.figure()
    fig, ax1 = plt.subplots(ncols=1, figsize=(16, 16))
    ax1.set_title('Original')
    #ax2.set_title('Attention Map Last Layer')
    _ = ax1.imshow(original_img)
    #_ = ax2.imshow(att_map)
    plt.savefig(os.path.join(output_location))
    fig.clf()
    plt.clf()
    plt.close("all")
    del att_map
    del original_img
    del fig
    gc.collect()



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


# generate spectrograms of some samples of the data
base_directory="../../dev_data/"
output_base_directory="./results/spectrograms_no_log_mel/"

temp = open("./results/spectrograms_no_log_mel/gearbox/temp.txt")

for machine in ['gearbox']:
    for domain in os.listdir(base_directory+"/"+machine):
        input_directory = base_directory + machine + "/" + domain
        output_directory = output_base_directory + machine+'/'+domain
        count = 0
        for filename in os.listdir(input_directory):
            if filename.endswith(".wav") and count % 100 == 0:
                file_location = os.path.join(input_directory, filename)
                sample_name = os.path.splitext(file_location[len(input_directory):])[0]
                output_location = output_directory + sample_name + ".png"
                convert_to_spectrogram_and_save(file_location, output_location,log_mel=False)
            count+=1

        print(machine+" "+domain+" done")


# generate attation maps of the samples

adast_mdl = adast()

start_interval=1
end_interval = start_interval+1

base_directory_spectrograms="./results/spectrograms/"
base_directory="../../dev_data/"
output_base_directory="./results/preproccesing_visualizations/"
skipcount=0
#for machine in os.listdir(base_directory_spectrograms):
for machine in ['gearbox']:
    for domain in ['train']:
        skipcount=0

        torch.cuda.empty_cache()
        print("cache cleared")
    #for domain in os.listdir(base_directory_spectrograms+"/"+machine):
        input_directory_specrograms = base_directory_spectrograms + machine + "/" + domain
        input_directory = base_directory+ machine + "/" + domain
        output_directory = output_base_directory + machine+'/'+domain
        for filename in os.listdir(input_directory_specrograms):
            if filename.endswith(".png") and skipcount>20 and skipcount<40 :
                print(skipcount)
                torch.cuda.empty_cache()
                print("cache cleared")

                wav_file_location = os.path.join(input_directory, filename[:-3] + "wav")
                png_file_location = os.path.join(input_directory_specrograms, filename)


                sample_name = os.path.splitext(wav_file_location[len(input_directory):])[0]
                output_location = output_directory + sample_name + "_attention_map.png"

                img=Image.open(png_file_location)
                save_spectrogram(img, img,output_location)
                gc.collect()
            skipcount+=1

        print(machine+" "+domain+" done")


