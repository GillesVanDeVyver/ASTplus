import os

import torch
import torchaudio
import matplotlib.pyplot as plt

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












# the rest of the cde is copy pasted from mobilenet baseline and slightly adjusted

########################################################################
# import python-library
########################################################################
# default
import glob
import argparse
import sys
import os
import itertools
import re

# additional
import numpy as np
import librosa
import librosa.core
import librosa.feature
import yaml

########################################################################


########################################################################
# setup STD I/O
########################################################################
"""
Standard output is logged in "baseline.log".
"""
import logging

logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

########################################################################


########################################################################
# version
########################################################################
__versions__ = "1.0.0"


########################################################################


########################################################################
# argparse
########################################################################
def command_line_chk():
    parser = argparse.ArgumentParser(description='Without option argument, it will not run properly.')
    parser.add_argument('-v', '--version', action='store_true', help="show application version")
    parser.add_argument('-d', '--dev', action='store_true', help="run mode Development")
    parser.add_argument('-e', '--eval', action='store_true', help="run mode Evaluation")
    args = parser.parse_args()
    if args.version:
        print("===============================")
        print("DCASE 2021 task 2 baseline\nversion {}".format(__versions__))
        print("===============================\n")
    if args.dev:
        flag = True
    elif args.eval:
        flag = False
    else:
        flag = None
        print("incorrect argument")
        print("please set option argument '--dev' or '--eval'")
    return flag


########################################################################


########################################################################
# load parameter.yaml
########################################################################
def yaml_load():
    with open("./baseline.yaml") as stream:
        param = yaml.safe_load(stream)
    return param


########################################################################


########################################################################
# file I/O
########################################################################
# wav file input
def file_load(wav_name, mono=False):
    """
    load .wav file.

    wav_name : str
        target .wav file
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for mono data

    return : numpy.array( float )
    """
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except:
        logger.error("file_broken or not exists!! : {}".format(wav_name))


########################################################################


########################################################################
# feature extractor
########################################################################
def file_to_vectors(file_name,
                    n_mels=64,
                    n_frames=5,
                    n_fft=1024,
                    hop_length=512,
                    power=2.0,
                    flatten=True):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    # calculate the number of dimensions
    dims = n_mels * n_frames

    # generate melspectrogram using librosa
    y, sr = file_load(file_name, mono=True)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)

    # convert melspectrogram to log mel energies
    log_mel_spectrogram = 20.0 / power * np.log10(np.maximum(mel_spectrogram, sys.float_info.epsilon))

    if flatten:
        # calculate total vector size
        n_vectors = len(log_mel_spectrogram[0, :]) - n_frames + 1

        # skip too short clips
        if n_vectors < 1:
            return np.empty((0, dims))

        # generate feature vectors by concatenating multiframes
        vectors = np.zeros((dims, n_vectors))
        for t in range(n_frames):
            vectors[n_mels * t: n_mels * (t + 1), :] = log_mel_spectrogram[:, t: t + n_vectors]
        return vectors

    else:
        return log_mel_spectrogram


########################################################################


########################################################################
# get directory paths according to mode
########################################################################
def select_dirs(param, mode):
    """
    param : dict
        baseline.yaml data

    return :
        if active type the development :
            dirs :  list [ str ]
                load base directory list of dev_data
        if active type the evaluation :
            dirs : list [ str ]
                load base directory list of eval_data
    """
    if mode:
        logger.info("load_directory <- development")
        query = os.path.abspath("{base}/*".format(base=param["dev_directory"]))
    else:
        logger.info("load_directory <- evaluation")
        query = os.path.abspath("{base}/*".format(base=param["eval_directory"]))
    dirs = sorted(glob.glob(query))
    dirs = [f for f in dirs if os.path.isdir(f)]
    return dirs


########################################################################


########################################################################
# get machine IDs
########################################################################
def get_section_names(target_dir,
                      dir_name,
                      ext="wav"):
    """
    target_dir : str
        base directory path
    dir_name : str
        sub directory name
    ext : str (default="wav)
        file extension of audio files

    return :
        section_names : list [ str ]
            list of section names extracted from the names of audio files
    """
    # create test files
    query = os.path.abspath("{target_dir}/{dir_name}/*.{ext}".format(target_dir=target_dir, dir_name=dir_name, ext=ext))
    file_paths = sorted(glob.glob(query))
    # extract section names
    section_names = sorted(list(set(itertools.chain.from_iterable(
        [re.findall('section_[0-9][0-9]', ext_id) for ext_id in file_paths]))))
    return section_names


########################################################################


########################################################################
# get the list of wave file paths
########################################################################
def file_list_generator(target_dir,
                        section_name,
                        dir_name,
                        mode,
                        prefix_normal="normal",
                        prefix_anomaly="anomaly",
                        ext="wav"):
    """
    target_dir : str
        base directory path
    section_name : str
        section name of audio file in <<dir_name>> directory
    dir_name : str
        sub directory name
    prefix_normal : str (default="normal")
        normal directory name
    prefix_anomaly : str (default="anomaly")
        anomaly directory name
    ext : str (default="wav")
        file extension of audio files

    return :
        if the mode is "development":
            files : list [ str ]
                audio file list
            labels : list [ boolean ]
                label info. list
                * normal/anomaly = 0/1
        if the mode is "evaluation":
            files : list [ str ]
                audio file list
    """
    logger.info("target_dir : {}".format(target_dir + "_" + section_name))

    # development
    if mode:
        query = os.path.abspath(
            "{target_dir}/{dir_name}/{section_name}_*_{prefix_normal}_*.{ext}".format(target_dir=target_dir,
                                                                                      dir_name=dir_name,
                                                                                      section_name=section_name,
                                                                                      prefix_normal=prefix_normal,
                                                                                      ext=ext))
        normal_files = sorted(glob.glob(query))
        normal_labels = np.zeros(len(normal_files))

        query = os.path.abspath(
            "{target_dir}/{dir_name}/{section_name}_*_{prefix_normal}_*.{ext}".format(target_dir=target_dir,
                                                                                      dir_name=dir_name,
                                                                                      section_name=section_name,
                                                                                      prefix_normal=prefix_anomaly,
                                                                                      ext=ext))
        anomaly_files = sorted(glob.glob(query))
        anomaly_labels = np.ones(len(anomaly_files))

        files = np.concatenate((normal_files, anomaly_files), axis=0)
        labels = np.concatenate((normal_labels, anomaly_labels), axis=0)

        logger.info("#files : {num}".format(num=len(files)))
        if len(files) == 0:
            logger.exception("no_wav_file!!")
        print("\n========================================")

    # evaluation
    else:
        query = os.path.abspath("{target_dir}/{dir_name}/{section_name}_*.{ext}".format(target_dir=target_dir,
                                                                                        dir_name=dir_name,
                                                                                        section_name=section_name,
                                                                                        ext=ext))
        files = sorted(glob.glob(query))
        labels = None
        logger.info("#files : {num}".format(num=len(files)))
        if len(files) == 0:
            logger.exception("no_wav_file!!")
        print("\n=========================================")

    return files, labels
########################################################################
