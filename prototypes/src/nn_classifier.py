import os
import numpy as np
import yaml
import pandas as pd
from ASD_ASTplus.Gilles.utilities import common
import csv
from ASD_ASTplus.Gilles.prototypes.src import keras_model_classifier
from ASD_ASTplus.Gilles.dcase2021_task2_baseline_mobile_net_v2 import common as com
import visualizer
import sys
from sklearn import metrics

try:
    from sklearn.externals import joblib
except:
    import joblib

# No finetuning yet

result_dir="./results/nn_classifier/single_layer/"
f = open(result_dir+"AUC_scores_single_layer.txt", 'w')
f_csv = open(result_dir+"AUC_scores_single_layer.csv", 'w')

csv_writer = csv.writer(f_csv)
header=["machine","AUC_source","pAUC_source","AUC_target","pAUC_target"]
csv_writer.writerow(header)


#skip_count=0
embedding_base_directory="../dev_data_embeddings/"

with open("nn_classifier.yaml") as stream:
    param = yaml.safe_load(stream)



def combine_into_dataframe(file_list,pickle_location):
    X = common.combine_embeddings_file_list(file_list)
    X.to_pickle(pickle_location)

for machine in os.listdir(embedding_base_directory):
    machine_dir=embedding_base_directory+machine
    csv_row=[]
    if os.path.isdir(machine_dir):
        for domain in ["train", "source_test", "target_test"]:
            section_names_file_path="./nn_classifier_models/section_names_"+machine+".pkl"
            print(machine)
            print(domain)
            section_names = com.get_section_names(machine_dir, dir_name=domain, ext="pt")
            nb_sections = len(section_names)
            joblib.dump(section_names, section_names_file_path)
            print("============== DATASET_GENERATOR ==============")
            nb_files_ea_section = []
            total_file_list=None
            first_section=True
            y_true_domain=None
            for section_idx, section_name in enumerate(section_names):
                print("section:"+section_name)

                # get file list for each section
                # all values of y_true are zero in training
                file_list, y_true = com.file_list_generator(target_dir=machine_dir,
                                                        section_name=section_name,
                                                        dir_name=domain,
                                                        mode=True,
                                                        ext="pt")
                if first_section:
                    total_file_list=file_list
                    y_true_domain=y_true
                    first_section=False
                else:
                    total_file_list=np.append(total_file_list,file_list)
                    y_true_domain=np.append(y_true_domain,y_true)

                nb_files_ea_section.append(len(file_list))
            pickle_location = machine_dir + "/" + domain + "/" + "dataframe_classifier.pkl"
            #combine_into_dataframe(total_file_list,pickle_location) # only first time needed

            X = pd.read_pickle(pickle_location)
            # make one-hot encoding for sections
            one_hot_labels_soft = np.full((len(X), nb_sections),param["fit"]["epsilon"],dtype=float)
            one_hot_labels_hard =np.zeros((len(X), nb_sections),dtype=float)
            i = 0
            for section_idx in range(nb_sections):
                one_hot_labels_soft[i:i+nb_files_ea_section[section_idx], section_idx] = 1-(nb_sections-1)*param["fit"]["epsilon"]
                one_hot_labels_hard[i:i+nb_files_ea_section[section_idx], section_idx] = 1
                i += nb_files_ea_section[section_idx]



            if domain=="train":
                print("============== MODEL TRAINING ==============")
                model = keras_model_classifier.denseNN(param["ast_embedding_size"], nb_sections, param["fit"]["lr"],
                                                       param["fit"]["nbLayers"], param["fit"]["normalization"],
                                                       param["fit"]["activation"])
                history_img_location = "./nn_classifier_models/history_"+machine
                model_file_path="./nn_classifier_models/model_"+machine
                model.summary()

                history = model.fit(x=X,
                                    y=one_hot_labels_soft,
                                    epochs=param["fit"]["epochs"],
                                    batch_size=param["fit"]["batch_size"],
                                    shuffle=param["fit"]["shuffle"],
                                    validation_split=param["fit"]["validation_split"],
                                    verbose=param["fit"]["verbose"])
                new_visualizer = visualizer.visualizer()
                new_visualizer.loss_plot(history.history["loss"], history.history["val_loss"])
                new_visualizer.save_figure(history_img_location)
                model.save(model_file_path)
            else:
                predictions = model.predict(X)
                anomaly_scores=[]
                for i in range(len(predictions)):
                    prediction = predictions[i]
                    true_section=np.nonzero(one_hot_labels_hard[i])[0][0]
                    prediction_true_section=prediction[true_section]
                    anomaly_scores.append(np.log(np.maximum(1.0 - prediction_true_section, sys.float_info.epsilon))
                                               - np.log(np.maximum(prediction_true_section, sys.float_info.epsilon)))
                print(domain)
                auc= metrics.roc_auc_score(y_true_domain, anomaly_scores)
                print(auc)
                pauc= metrics.roc_auc_score(y_true_domain, anomaly_scores, max_fpr=param["max_fpr"])
                print(pauc)

                fpr_target, tpr_target, thresholds_target = metrics.roc_curve(y_true_domain, anomaly_scores)

                ROC_location_target = result_dir + "ROC_target_"+machine+"_"+domain+".png"
                common.generate_ROC_curve(fpr_target, tpr_target, ROC_location_target)


                if domain=="source_test":
                    f.write(str(machine) + ":\n" +
                            "AUC source test=" + str(auc) + ", pAUC target test=" + str(pauc) + "\n")
                    csv_row = [machine, auc, pauc]
                else:
                    f.write("AUC target test=" + str(auc) + ", pAUC target test=" + str(pauc) + "\n")
                    csv_row.extend([auc,pauc])
                    csv_writer.writerow(csv_row)



















