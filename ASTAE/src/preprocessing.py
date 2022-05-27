import os
import common
import torch
import pandas as pd
import re

# compute and save spectrograms of input data
def generate_spectrograms(input_base_directory,output_base_directory):
    for machine in os.listdir(input_base_directory):
        for domain in os.listdir(input_base_directory+"/"+machine):
            input_directory = input_base_directory + machine + "/" + domain
            output_directory = output_base_directory + machine+'/'+domain
            for filename in os.listdir(input_directory):
                if filename.endswith(".wav"):
                    file_location = os.path.join(input_directory, filename)
                    sample_name = os.path.splitext(file_location[len(input_directory):])[0]
                    output_location = output_directory + sample_name + ".pt"
                    log_mel = common.convert_to_log_mel(file_location)
                    torch.save(log_mel, output_location)
            print(machine+" "+domain+" done")



def generate_dataframes(input_base_directory,output_base_directory,sample=False):
    #for machine in os.listdir(input_base_directory):
    if sample:
        list_of_machines=['debug_sample']
    else:
        list_of_machines = ["gearbox","fan","pump","slider","ToyCar","ToyTrain","valve"]
    for machine in list_of_machines:


        for domain in os.listdir(input_base_directory+"/"+machine):
            tensors_in_domain = None
            lables = []
            index_lables=[]
            print("starting " + machine+" "+domain)
            input_directory = input_base_directory + machine + "/" + domain
            output_directory = output_base_directory + machine+'/'+domain+"/"
            for filename in os.listdir(input_directory):
                if filename.endswith(".pt"):
                    file_location = os.path.join(input_directory, filename)
                    if "anomaly" in filename:
                        lables.append(1)
                    else:
                        lables.append(0)
                    section_as_str = filename[9]
                    section = int(section_as_str)
                    index_lables.append(section)


                    loaded_tensor = torch.unsqueeze(torch.load(file_location),0)
                    if tensors_in_domain == None:
                        tensors_in_domain = loaded_tensor
                    else:
                        tensors_in_domain = torch.cat((tensors_in_domain, loaded_tensor))
            output_location_dataframe = output_directory + "dataframe.pt"
            output_location_labels = output_directory + "labels.pt"
            output_location_indices = output_directory + "index_labels.pt"

            #px = pd.DataFrame(tensors_in_domain.detach().numpy())
            #px.to_pickle(output_location_dataframe)
            torch.save(lables, output_location_labels)
            torch.save(index_lables, output_location_indices)
            torch.save(tensors_in_domain.detach(),output_location_dataframe)

            print(machine+" "+domain+" done")




server = False
sample=False


if sample:
    raw_audio_base_directory="../../dev_sample/"
else:
    raw_audio_base_directory="../../dev_data/"



if server:
    spectrograms_base_directory="../../dev_data_spectrograms_server/"
else:
    if sample:
        spectrograms_base_directory="../../dev_sample_spectrograms/"
    else:
        spectrograms_base_directory="../../dev_data_spectrograms/"


if server:
    dataframes_base_directory="../../dev_data_dataframes_server/"
else:
    if sample:
        dataframes_base_directory="../../dev_sample_dataframes/"
    else:
        dataframes_base_directory="../../dev_data_dataframes/"

#generate_spectrograms(raw_audio_base_directory,spectrograms_base_directory)
generate_dataframes(spectrograms_base_directory,dataframes_base_directory,sample=sample)
#test=torch.load("../../dev_sample_dataframes/debug_sample/train/dataframe.pt")
#test2=torch.load("../../dev_data_dataframes/fan/train/dataframe.pt")
#print(test2)