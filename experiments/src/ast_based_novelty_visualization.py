import os, sys
parentdir = str(os.path.abspath(os.path.join(__file__, "../../../../"))) + '/src'
print(parentdir)
sys.path.append(parentdir)
import models
import torch
import common
import pandas as pd


class adast(): # anomaly detection ast_model
    def __init__(self,input_tdim = 1024,num_mel_bins=128,embedding_dimension=768):
        # audioset input sequence length is 1024
        pretrained_mdl_path = '../../pretrained_models/audioset_10_10_0.4593.pth'
        # get the frequency and time stride of the pretrained model from its name
        fstride, tstride = int(pretrained_mdl_path.split('/')[-1].split('_')[1]), int(pretrained_mdl_path.split('/')[-1].split('_')[2].split('.')[0])
        # The input of audioset pretrained model is 1024 frames.
        self.input_tdim = input_tdim
        # initialize an AST model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #device = torch.device("cuda")
        torch.cuda.empty_cache()
        print("cache cleared")
        sd = torch.load(pretrained_mdl_path, map_location=device)
        audio_model_ast = models.ASTModel(input_tdim=input_tdim, fstride=fstride, tstride=tstride)
        self.audio_model = torch.nn.DataParallel(audio_model_ast)
        self.audio_model.load_state_dict(sd, strict=False)
        self.num_mel_bins=num_mel_bins
        self.embedding_dimension=embedding_dimension


    def get_ast_embedding_single_file(self,file_location):
        log_mel = common.convert_to_log_mel(file_location, num_mel_bins=self.num_mel_bins, target_length=self.input_tdim)
        input = torch.unsqueeze(log_mel, dim=0)
        #input = torch.rand([1, self.input_tdim, 128])
        output = self.audio_model(input)
        return output


    def generate_and_save_embeddings(self,input_location,output_location):
        output = self.get_ast_embedding_single_file(input_location)
        torch.save(output,output_location)

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

def generate_lables_and_pd_dataframe(input_directory,format="one_class_svm"):
    tensors_in_domain = None
    lables = []
    for filename in os.listdir(input_directory):
        if filename.endswith(".pt"):
            if "anomaly" in filename:
                if format=="one_class_svm":
                    lables.append(-1)
                elif format=="autoencoder":
                    lables.append(1)
            else:
                if format=="one_class_svm":
                    lables.append(1)
                elif format=="autoencoder":
                    lables.append(0)
            file_location = input_directory + "/" + filename
            loaded_tensor = torch.load(file_location)
            if tensors_in_domain == None:
                tensors_in_domain = loaded_tensor
            else:
                tensors_in_domain = torch.cat((tensors_in_domain, loaded_tensor))
    px = pd.DataFrame(tensors_in_domain.detach().numpy())
    return lables,px


# generate intermediate tensors and store as .pt files
"""
adast_mdl = adast()
base_directory="../../dev_data/"
output_base_directory="../dev_data_embeddings/"
for machine in os.listdir(base_directory):
    for domain in os.listdir(base_directory+"/"+machine):
        input_directory = base_directory + machine + "/" + domain
        output_directory = output_base_directory + machine+'/'+domain
        for filename in os.listdir(input_directory):
            if filename.endswith(".wav"):
                file_location = os.path.join(input_directory, filename)
                sample_name = os.path.splitext(file_location[len(input_directory):])[0]
                output_location = output_directory + sample_name + ".pt"
                adast_mdl.generate_and_save_embeddings(file_location, output_location)
        print(machine+" "+domain+" done")

"""






#combine individual tensors stored in .pt to one pandas dataframe and save as pkl file => not enough RAM
"""
embedding_base_directory="../dev_data_embeddings/"
for machine in os.listdir(embedding_base_directory):
    if os.path.isdir(embedding_base_directory+"/"+machine):
        print(machine)
        for domain in os.listdir(embedding_base_directory + "/" + machine):
            machine_dir=embedding_base_directory + machine
            input_directory = embedding_base_directory + machine + "/" + domain
            if os.path.isdir(input_directory):
                X=load_embeddings(input_directory)
                pickle_location=input_directory+"/"+"dataframe.pkl"
                X.to_pickle(pickle_location)
                print(X.shape)
        print(machine+" "+domain+" done")

"""










