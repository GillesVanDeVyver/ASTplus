import os
import numpy as np
import yaml
import pandas as pd
from ASD_ASTplus.Gilles.utilities import common
import csv
from ASD_ASTplus.Gilles.dcase2021_task2_baseline_mobile_net_v2 import common as com
from sklearn import metrics
from sklearn import svm
try:
    from sklearn.externals import joblib
except:
    import joblib

# No finetuning yet

result_dir="./results/svm_classifier/"
f = open(result_dir+"AUC_scores_SVM.txt", 'w')
f_csv = open(result_dir+"AUC_scores_SVM.csv", 'w')

csv_writer = csv.writer(f_csv)
header=["machine","AUC_source","pAUC_source","AUC_target","pAUC_target"]
csv_writer.writerow(header)


#skip_count=0
embedding_base_directory="../dev_data_embeddings/"

with open("svm_classifier.yaml") as stream:
    param = yaml.safe_load(stream)



def combine_into_dataframe(file_list,pickle_location):
    X = common.combine_embeddings_file_list(file_list)
    X.to_pickle(pickle_location)

for machine in os.listdir(embedding_base_directory):
    machine_dir=embedding_base_directory+machine
    csv_row=[]
    if os.path.isdir(machine_dir):
        for domain in ["train", "source_test", "target_test"]:
            section_names_file_path="./nn_classifier_models/section_names_"+machine+".pkl" #reuse section section names
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
            labels = np.full((len(X)),param["fit"]["epsilon"],dtype=float)
            i = 0
            for section_idx in range(nb_sections):
                labels[i:i+nb_files_ea_section[section_idx]] = section_idx
                i += nb_files_ea_section[section_idx]




            if domain=="train":
                print("============== MODEL TRAINING ==============")
                model = svm.SVC()
                history = model.fit(X,labels)
            else:
                predictions = model.predict(X)
                anomaly_scores=[]
                for i in range(len(predictions)):
                    prediction = predictions[i]
                    true_section=labels[i]
                    if abs(prediction-true_section)<0.1:
                        anomaly_scores.append(0)
                    else:
                        anomaly_scores.append(1)

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
                            "AUC source test=" + str(auc) + ", pAUC source test=" + str(pauc) + "\n")
                    csv_row = [machine, auc, pauc]
                else:
                    f.write("AUC target test=" + str(auc) + ", pAUC target test=" + str(pauc) + "\n")
                    csv_row.extend([auc,pauc])
                    csv_writer.writerow(csv_row)



















