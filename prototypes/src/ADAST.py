import os, sys
parentdir = str(os.path.abspath(os.path.join(__file__, "../../../"))) + '/src'
print(parentdir)
sys.path.append(parentdir)
import models
import torch
grandparentdir = str(os.path.abspath(os.path.join(__file__, "../../../../../")))
print(grandparentdir)
sys.path.append(grandparentdir)
import common
import pandas as pd

# ADAST: Anomaly Detector AST
class adast(): # anomaly detection ast_model
    def __init__(self,input_tdim = 1024,num_mel_bins=128,embedding_dimension=768,full_output=False,number_of_layers=None):
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
        print(device)
        sd = torch.load(pretrained_mdl_path, map_location=device)
        if full_output:
            audio_model_ast = models.ASTModel_full(input_tdim=input_tdim, fstride=fstride, tstride=tstride,number_of_layers=number_of_layers)
        else:
            audio_model_ast = models.ASTModel(input_tdim=input_tdim, fstride=fstride, tstride=tstride,number_of_layers=number_of_layers)
        self.audio_model = torch.nn.DataParallel(audio_model_ast)
        self.audio_model.load_state_dict(sd, strict=False)
        self.num_mel_bins=num_mel_bins
        self.embedding_dimension=embedding_dimension


    def get_ast_embedding_single_file(self,file_location,device):
        log_mel = common.convert_to_log_mel(file_location, num_mel_bins=self.num_mel_bins, target_length=self.input_tdim)
        input = torch.unsqueeze(log_mel, dim=0)
        #input = torch.rand([1, self.input_tdim, 128])
        input=input.to(device)
        self.audio_model=self.audio_model.to(device)
        output = self.audio_model(input)
        return output


    def generate_and_save_embeddings(self,input_location,output_location,device):
        output = self.get_ast_embedding_single_file(input_location,device)
        torch.save(output,output_location)



def generate_lables_and_pd_dataframe(input_directory,format="one_class_svm",custom_label=None):
    tensors_in_domain = None
    lables = []
    for filename in os.listdir(input_directory):
        if filename.endswith(".pt"):
            if "anomaly" in filename:
                if format=="one_class_svm":
                    lables.append(-1)
                elif format=="autoencoder":
                    lables.append(1)
                elif format == "inter_machine_class":
                    lables.append(custom_label)
            else:
                if format=="one_class_svm":
                    lables.append(1)
                elif format=="autoencoder":
                    lables.append(0)
                elif format == "inter_machine_class":
                    lables.append(custom_label)
            file_location = input_directory + "/" + filename
            loaded_tensor = torch.load(file_location)
            if tensors_in_domain == None:
                tensors_in_domain = loaded_tensor
            else:
                loaded_tensor.to("cpu")
                tensors_in_domain.to('cpu')
                tensors_in_domain = torch.cat((tensors_in_domain, loaded_tensor))
    px = pd.DataFrame(tensors_in_domain.detach().cpu().numpy())
    return lables,px


# generate intermediate tensors and store as .pt files
full_output = False
server = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nb_layers = 5
adast_mdl = adast(full_output=full_output,number_of_layers=nb_layers)
base_directory="../../dev_data/"


if nb_layers ==None:
    if full_output:
        output_base_directory="../../dev_data_embeddings_full/"
    else:
        output_base_directory="../../dev_data_embeddings/"
else:
    if full_output:
        output_base_directory="../../dev_data_embeddings_full_"+str(nb_layers)+"layers/"
    else:
        output_base_directory="../../dev_data_embeddings_"+str(nb_layers)+"layers/"




for machine in ["gearbox","valve","slider","ToyTrain","fan","pump","ToyCar"]:
    for domain in os.listdir(base_directory+"/"+machine):
        input_directory = base_directory + machine + "/" + domain
        output_directory = output_base_directory + machine+'/'+domain
        for filename in os.listdir(input_directory):
            if filename.endswith(".wav"):
                file_location = os.path.join(input_directory, filename)
                sample_name = os.path.splitext(file_location[len(input_directory):])[0]
                output_location = output_directory + sample_name + ".pt"
                adast_mdl.generate_and_save_embeddings(file_location, output_location,device)
        print(machine+" "+domain+" done")





# save as dataframe

#embedding_base_directory="../../dev_data_embeddings/"

embedding_base_directory=output_base_directory
for machine in ["gearbox","valve","slider","ToyTrain","fan","pump","ToyCar"]:
    if os.path.isdir(embedding_base_directory+"/"+machine):
        print(machine)
        for domain in os.listdir(embedding_base_directory + "/" + machine):
            machine_dir=embedding_base_directory + machine
            input_directory = embedding_base_directory + machine + "/" + domain
            if os.path.isdir(input_directory):
                X=common.load_embeddings(input_directory)
                pickle_location=input_directory+"/"+"dataframe.pkl"
                X.to_pickle(pickle_location)
                print(X.shape)
        print(machine+" "+domain+" done")


#embedding_base_directory = "../dev_data_embeddings/"
#common.combine_embeddings(embedding_base_directory)







