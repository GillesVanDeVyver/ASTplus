import torch
import torchaudio
import matplotlib.pyplot as plt
import os, sys
import pandas as pd

def convert_to_log_mel(path_to_file, num_mel_bins=128, target_length=1024):
    waveform, sample_rate = torchaudio.load(path_to_file)
    fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sample_rate, use_energy=False,
                                              window_type='hanning', num_mel_bins=num_mel_bins, dither=0.0,
                                              frame_shift=10)  # using parameters from AST
    n_frames = fbank.size(dim=0)
    p = target_length - n_frames
    # cut and pad
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]
    return fbank





def generate_confusion_matrix(prediction_labels, true_labels):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    for i in range(len(prediction_labels)):
        if prediction_labels[i] == 1:
            if true_labels[i] == 1:
                true_neg = true_neg + 1
            else:
                false_neg = false_neg + 1
        else:
            if true_labels[i] == -1:
                true_pos = true_pos + 1
            else:
                false_pos = false_pos + 1
    return true_pos, true_neg, false_pos, false_neg

#uses trapezoidal rule
def calculate_pAUC_approximation(FPR_TRPs,p=1):
    a=0
    fa=0
    auc_approx=0
    for pair in FPR_TRPs:
        b=pair[0]
        fb=pair[1]
        if b<p:
            auc_approx=auc_approx+(b-a)*(fa+fb)/2
            a=b
            fa=b
    b=p
    fb=p
    auc_approx = auc_approx + (b - a) * (fa + fb) / 2
    return auc_approx/(p)

def generate_ROC_curve(FPR_TRPs,output_location):
    x_axis=[0]
    y_axis=[0]
    for pair in FPR_TRPs:
        x_axis.append(pair[0])
        y_axis.append(pair[1])
    x_axis.append(1)
    y_axis.append(1)
    plt.plot(x_axis,y_axis)
    plt.plot([0,1],[0,1])
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.savefig(output_location)
    plt.close()

def generate_ROC_curve(FPR,TRP,output_location):
    x_axis=[0]
    y_axis=[0]
    for i in range(len(FPR)):
        x_axis.append(FPR[i])
        y_axis.append(TRP[i])
    x_axis.append(1)
    y_axis.append(1)
    plt.plot(x_axis,y_axis)
    plt.plot([0,1],[0,1])
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.savefig(output_location)
    plt.close()

def combine_embeddings(embedding_base_directory):
    for machine in os.listdir(embedding_base_directory):
        if os.path.isdir(embedding_base_directory + "/" + machine):
            print(machine)
            for domain in os.listdir(embedding_base_directory + "/" + machine):
                machine_dir = embedding_base_directory + machine
                input_directory = embedding_base_directory + machine + "/" + domain
                if os.path.isdir(input_directory):
                    X = load_embeddings(input_directory)
                    pickle_location = input_directory + "/" + "dataframe.pkl"
                    X.to_pickle(pickle_location)
                    print(X.shape)
            print(machine + " " + domain + " done")

def combine_embeddings_file_list(file_list):
    concatenated_tensors = None
    for file in file_list:
        if file.endswith(".pt"):
            try:
                loaded_tensor = torch.load(file,map_location=torch.device('cpu'))
            except:
                loaded_tensor = torch.load(file)
            if concatenated_tensors == None:
                concatenated_tensors = loaded_tensor
            else:
                concatenated_tensors = torch.cat((concatenated_tensors, loaded_tensor))
    px = pd.DataFrame(concatenated_tensors.detach().numpy())
    return px

def load_embeddings(input_directory):
    tensors_in_domain = None
    for filename in os.listdir(input_directory):
        if filename.endswith(".pt"):
            file_location = input_directory + "/" + filename
            loaded_tensor = torch.load(file_location)
            if tensors_in_domain == None:
                tensors_in_domain = loaded_tensor
            else:
                tensors_in_domain = torch.cat((tensors_in_domain, loaded_tensor))
    px = pd.DataFrame(tensors_in_domain.detach().numpy())
    return px