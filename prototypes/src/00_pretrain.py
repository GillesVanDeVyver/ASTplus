########################################################################
# import default libraries
########################################################################
import os
import sys
import gc
########################################################################


########################################################################
# import additional libraries
########################################################################
import numpy as np
import scipy.stats
# from import
import torch
from tqdm import tqdm
try:
    from sklearn.externals import joblib
except:
    import joblib
# original lib
import common as com
import keras_model
import keras_model_pretrain
import sklearn

import keras
import tensorflow as tf

########################################################################


########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
########################################################################


########################################################################
# visualizer
########################################################################
class visualizer(object):
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = self.plt.figure(figsize=(7, 5))
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def loss_plot(self, loss, val_loss):
        """
        Plot loss curve.

        loss : list [ float ]
            training loss time series.
        val_loss : list [ float ]
            validation loss time series.

        return   : None
        """
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Validation"], loc="upper right")

    def save_figure(self, name):
        """
        Save figure.

        name : str
            save png file path.

        return : None
        """
        self.plt.savefig(name)


########################################################################


########################################################################
# get data from the list for file paths
########################################################################
def file_list_to_data(file_list,
                      start_idx,
                      end_idx,
                      msg="calc...",
                      n_mels=64,
                      n_frames=5,
                      n_hop_frames=1,
                      n_fft=1024,
                      hop_length=512,
                      power=2.0,):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.

    return : numpy.array( numpy.array( float ) )
        data for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    # calculate the number of dimensions
    dims = n_mels * n_frames

    # iterate file_to_vector_array()
    for idx in tqdm(range(start_idx,end_idx), desc=msg):
        vectors = com.file_to_vectors(file_list[idx],
                                                n_mels=n_mels,
                                                n_frames=n_frames,
                                                n_fft=n_fft,
                                                hop_length=hop_length,
                                                power=power)
        vectors = vectors[: : n_hop_frames, :]
        if idx == 0:
            data = np.zeros(((end_idx-start_idx) * vectors.shape[0], dims), float)
        data[vectors.shape[0] * idx : vectors.shape[0] * (idx + 1), :] = vectors

    return data


def file_list_to_AST_data(file_list,start_idx,end_idx,
                      msg="calc...",full_output=False):

    data = []
    for file in tqdm((file_list[start_idx:end_idx]), desc=msg):
        if full_output:
            raise ("not implemented yet")
        else:
            tensor = torch.load(file)
            data.append(tensor)
    return data

def format_full_data(ast_data,msg="calc...",n_vectors=309):
    dim = ast_data.size(1)
    for i in tqdm(range(len(ast_data)), desc=msg):
        tensor = ast_data[i]
        vectors = tensor.repeat((n_vectors,1))
        if i == 0:
            data = np.zeros((len(ast_data) * vectors.shape[0], dim), float)
    data[vectors.shape[0] * idx : vectors.shape[0] * (idx + 1), :] = vectors.cpu().detach().numpy()
    return data


"""
def format_single_tensor_ast(tensor,n_vectors,n_slices,n_frames,n_vectors_ast = 101,nb_patches_freq = 12):
    unflattened_tensor = tensor.unflatten(0,(nb_patches_freq,n_vectors_ast))
    vectors = []
    for i in range(n_vectors):
        offset = i+n_frames/2
        offset_ast = round(n_vectors_ast/n_slices*offset)
        corresponding_tensors = unflattened_tensor[:,offset_ast,:]
        stacked_embeds = corresponding_tensors.flatten()

        pca_embeds = sklearn.decomposition.PCA()

        vectors.append(stacked_embeds)
"""





########################################################################


########################################################################
# main 00_pretrain.py
########################################################################
full_embedding_location="../../dev_data_embeddings"
#split_size = 10
debug = False
if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False

    #PRETRAIN PART


    mode = com.command_line_chk()
    if mode is None:
        sys.exit(-1)
        
    # make output directory
    os.makedirs(param["model_directory"], exist_ok=True)

    # initialize the visualizer
    visualizer_pretrain = visualizer()

    # load base_directory list
    dirs = com.select_dirs(param=param, mode=mode)

    # loop of the base directory
    #skip_count = 0
    for idx, input_dir in enumerate(dirs):
        #if skip_count>3:
        print("\n===========================")
        print("[{idx}/{total}] {target_dir}".format(target_dir=input_dir, idx=idx+1, total=len(input_dir)))

        # set path
        machine_type = os.path.split(input_dir)[1]
        if machine_type!= 'fan':
            pass
        else:
            model_file_path = "{model}/model_{machine_type}.hdf5".format(model=param["model_directory"],
                                                                         machine_type=machine_type)

            #if os.path.exists(model_file_path):
                #com.logger.info("model exists")
                #continue

            history_img = "{model}/history_{machine_type}.png".format(model=param["model_directory"],
                                                                      machine_type=machine_type)
            # pickle file for storing anomaly score distribution
            score_distr_file_path = "{model}/score_distr_{machine_type}.pkl".format(model=param["model_directory"],
                                                                                    machine_type=machine_type)

            # generate dataset
            print("============== DATASET_GENERATOR ==============")


            print("============== INPUT DATA ==============")


            # get file list for all sections
            # all values of y_true are zero in training

            files, y_true = com.file_list_generator(target_dir=input_dir,
                                        section_name="*",
                                        dir_name="train",
                                        mode=mode) # input data
            pretrain_dir = full_embedding_location + "/"+machine_type

            ast_files, y_true = com.file_list_generator(target_dir=pretrain_dir,
                                        section_name="*",
                                        dir_name="train",
                                        mode=mode,
                                        ext="pt") #AST data

            nb_files = len(files)
            #split_idx = 0

            #while (split_idx<=nb_files):
            #print("split index: " + str(split_idx))
            #end_idx = split_idx+split_size
            #if end_idx > nb_files:
            #    end_idx=nb_files
            if debug:
                end_idx = 10
            else:
                end_idx=len(files)

            input_data = file_list_to_data(files,0,end_idx,
                                     msg="generate train_dataset",
                                     n_mels=param["feature"]["n_mels"],
                                     n_frames=param["feature"]["n_frames"],
                                     n_hop_frames=param["feature"]["n_hop_frames"],
                                     n_fft=param["feature"]["n_fft"],
                                     hop_length=param["feature"]["hop_length"],
                                     power=param["feature"]["power"])

            # number of vectors for each wave file
            n_vectors_ea_file = int(input_data.shape[0] / (end_idx-0))

            print("============== AST DATA ==============")




            data_type = os.path.split(input_dir)[0]






            ast_data = file_list_to_AST_data(ast_files,0,end_idx,
                                     msg="generate pretrain_dataset")

            # number of vectors for each wave file

            ast_dim = len(ast_data[0][0])

            for i in range(len(ast_data)):
                ast_data[i] = torch.squeeze(ast_data[i])
            ast_data = torch.stack(ast_data)

            ast_data=format_full_data(ast_data)





            # pretrain model
            print("============== MODEL TRAINING ==============")
            model = keras_model_pretrain.get_model(param["feature"]["n_mels"] * param["feature"]["n_frames"],
                                          param["pretrain"]["lr"],ast_dim)

            model.summary()

            history = model.fit(x=input_data,
                                y=ast_data,
                                epochs=param["pretrain"]["epochs"],
                                batch_size=param["pretrain"]["batch_size"],
                                shuffle=param["pretrain"]["shuffle"],
                                validation_split=param["pretrain"]["validation_split"],
                                verbose=param["pretrain"]["verbose"])

                #split_idx=end_idx



            # calculate y_pred for fitting anomaly score distribution
            y_pred = []
            start_idx = 0
            for file_idx in range(start_idx,end_idx):
                    y_pred.append(np.mean(np.square(ast_data[start_idx : start_idx + n_vectors_ea_file, :]
                                          - model.predict(input_data[start_idx : start_idx + n_vectors_ea_file, :]))))
                    start_idx += n_vectors_ea_file

            # fit anomaly score distribution
            shape_hat, loc_hat, scale_hat = scipy.stats.gamma.fit(y_pred)
            gamma_params = [shape_hat, loc_hat, scale_hat]
            joblib.dump(gamma_params, score_distr_file_path)
            visualizer_pretrain.loss_plot(history.history["loss"],history.history["val_loss"])
            visualizer_pretrain.save_figure(history_img)
            model.save(model_file_path)
            com.logger.info("save_model -> {}".format(model_file_path))
            print("============== END TRAINING ==============")

            del ast_data
            del input_data
            del model
            keras_model.clear_session()
            gc.collect()
