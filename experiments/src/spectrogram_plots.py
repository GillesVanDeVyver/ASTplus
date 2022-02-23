import common as com
import os
import numpy as np
from PIL import Image
import librosa.display
import matplotlib.pyplot as plt

#source
#sample_name="section_00_source_train_normal_0000_A1Spd6Mic1"
#sample_name="section_00_source_train_normal_0001_A1Spd6Mic1"
#sample_name="section_00_source_train_normal_0319_A2Spd8Mic1"

#target train
sample_name="section_00_target_train_normal_0002_C2Spd6Mic1"
sample_name="section_00_target_train_normal_0001_C1Spd8Mic1"
sample_name="section_00_target_train_normal_0000_C1Spd6Mic1"

sample_name="section_00_target_test_anomaly_0000"
sample_name="section_00_target_test_anomaly_0001"
sample_name="section_00_target_test_anomaly_0071"
#sample_name="section_00_target_test_normal_0028"

sample_name="v011z-c62m4"
sample_name="environmental"

sample_name="section_00_source_train_normal_0000_serial_no_000_water"
sample_name="section_00_source_train_normal_0000_0_g_25_mm_2000_mV_none"
sample_name="section_01_target_test_anomaly_0071"
param = com.yaml_load()

file_location=os.path.join("../dev_data/ToyTrain/train/"+sample_name+".wav")
file_location=os.path.join("../dev_data/ToyTrain/target_test/"+sample_name+".wav")

file_location=os.path.join("./extra data/"+sample_name+".wav")
file_location=os.path.join("../dev_data/pump/train/"+sample_name+".wav")
file_location=os.path.join("../devsection_01_source_train_normal_0647_120_g_20_mm_2000_mV_none_data/slider/train/"+sample_name+".wav")
file_location=os.path.join("../dev_data/gearbox/target_test/"+sample_name+".wav")



def convert_to_spectrogram_and_save(file_location,output_location):

    vectors = com.file_to_vectors(file_location,
                                  n_mels=param["feature"]["n_mels"],
                                  n_frames=param["feature"]["n_frames"],
                                  n_fft=param["feature"]["n_fft"],
                                  hop_length=param["feature"]["hop_length"],
                                  power=param["feature"]["power"],
                                  flatten=True)

    #print(np.shape(vectors[:,0]))
    log_mel_spectrogram = com.file_to_vectors(file_location,
                                  n_mels=param["feature"]["n_mels"],
                                  n_frames=param["feature"]["n_frames"],
                                  n_fft=param["feature"]["n_fft"],
                                  hop_length=param["feature"]["hop_length"],
                                  power=param["feature"]["power"],
                                  flatten=False)
    #print(log_mel_spectrogram[0][0])
    #print(np.shape(log_mel_spectrogram))



    plt.figure()

    plt.axis('off')  # no axis
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge
    librosa.display.specshow(log_mel_spectrogram)
    plt.savefig(os.path.join(output_location))
    #plt.show()
    plt.close()
"""
plt.figure()
librosa.display.specshow(vectors)
plt.savefig(os.path.join("results/spectrograms/"+sample_name+"_vectors"))
plt.close()
"""