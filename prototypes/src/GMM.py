import os
import numpy as np
import yaml as yaml
import ADAST
import pandas as pd
import common
import csv
from sklearn import metrics
from pathlib import Path
from sklearn import mixture
from sklearn.decomposition import PCA

accuracies = [[] for i in range(7)]

print(accuracies)

min_nb_comp=1
max_nb_comp=1
step=1

nb_pca_comps=300

#for nb_comp in range(min_nb_comp,max_nb_comp+1,step):

nb_comp = 1

machine_offset = 0
with open("GMM.yaml") as stream:
    param = yaml.safe_load(stream)
result_dir="./results/GMMv2/nb_comp="+str(nb_comp)+"_n_init="+\
                             str(param["fit"]["n_init"])+"_covariance_type="+\
                             param["fit"]["cov_type"]+"/"
Path(result_dir).mkdir(parents=True, exist_ok=True)
f = open(result_dir+"AUC_scores_GMM_nb_comp="+str(nb_comp)+"_n_init="+\
                             str(param["fit"]["n_init"])+"_covariance_type="+ \
                             #"pca_components = " + str(nb_pca_comps) + \
                             param["fit"]["cov_type"]+".txt", 'w')
f_csv = open(result_dir+"AUC_scores_GMM_nb_comp="+str(nb_comp)+"_n_init="+\
                             str(param["fit"]["n_init"])+"_covariance_type="+ \
                             #"pca_components = " + str(nb_pca_comps) + \
                             param["fit"]["cov_type"]+".csv", 'w')

csv_writer = csv.writer(f_csv)
header=["machine","AUC_source","pAUC_source","AUC_target","pAUC_target"]
csv_writer.writerow(header)
embedding_base_directory="../../dev_data_embeddings_5layers/"
for machine in ["gearbox","valve","slider","ToyTrain","fan","pump","ToyCar"]:
    if os.path.isdir(embedding_base_directory+"/"+machine):
        machine_dir = embedding_base_directory + machine
        train_pickle_location = machine_dir + "/train/" + "dataframe.pkl"

        lables_train, X = ADAST.generate_lables_and_pd_dataframe(
            machine_dir + "/train/",format="autoencoder")
        X = pd.read_pickle(train_pickle_location)

        model = mixture.GaussianMixture(n_components= nb_comp,n_init=param["fit"]["n_init"],
                                        covariance_type=param["fit"]["cov_type"])
        #pca = PCA(n_components=nb_pca_comps)
        #X_pca = pca.fit_transform(X)
        model.fit(X)
        # source prediction
        labels_source_test, X_source_test = ADAST.generate_lables_and_pd_dataframe(
            machine_dir + "/source_test",format="autoencoder")
        #X_source_test_pca=pca.fit_transform(X_source_test)
        source_prediction = -model.score_samples(X_source_test) # higher value means more normal => - for anomaly score
        auc_source = metrics.roc_auc_score(labels_source_test, source_prediction)
        pauc_source= metrics.roc_auc_score(labels_source_test, source_prediction,max_fpr=param["max_fpr"])
        fpr_source, tpr_source, thresholds_source = metrics.roc_curve(labels_source_test, source_prediction)

        ROC_location_source=result_dir+"ROC_source_"+str(machine)+"_nb_comp="+str(nb_comp)+"_n_init="+\
                             str(param["fit"]["n_init"])+"_covariance_type="+param["fit"]["cov_type"]+".png"
                            #"pca_components=" + str(nb_pca_comps) +\

        common.generate_ROC_curve(fpr_source, tpr_source, ROC_location_source)
        # target prediction
        labels_target_test, X_target_test = ADAST.generate_lables_and_pd_dataframe(
            machine_dir + "/target_test",format="autoencoder")
        #X_target_test_pca = pca.fit_transform(X_target_test)
        target_prediction = -model.score_samples(X_target_test)
        auc_target= metrics.roc_auc_score(labels_target_test, target_prediction)
        pauc_target= metrics.roc_auc_score(labels_target_test, target_prediction,max_fpr=param["max_fpr"])
        fpr_target, tpr_target, thresholds_target = metrics.roc_curve(labels_target_test, target_prediction)
        ROC_location_target=result_dir+"ROC_target_"+str(machine)+"_nb_comp="+str(nb_comp)+"_n_init="+\
                             str(param["fit"]["n_init"])+"_covariance_type="+param["fit"]["cov_type"]+".png"
                             #"pca_components = " + str(nb_pca_comps) +\

        common.generate_ROC_curve(fpr_target, tpr_target, ROC_location_target)
        f.write(str(machine) + ":\n"+
                "AUC source test="  + str(auc_source) + ", pAUC source test=" + str(pauc_source) + "\n"+
                "AUC target test=" + str(auc_target) + ", pAUC target test=" + str(pauc_target) +  "\n")

        csv_row = [machine, auc_source, pauc_source, auc_target,pauc_target]
        csv_writer.writerow(csv_row)
        new_accuracies=[nb_comp,auc_source,pauc_source,auc_target,pauc_target]
        accuracies[machine_offset].append(new_accuracies)
        machine_offset+=1
        output_loc_accuracies = "./results/GMMv2/accuracies_covtype=" + param["fit"]["cov_type"] + \
                                "_nbinit=" + str(param["fit"]["n_init"]) + \
                                "[" + str(min_nb_comp) + ":" + \
                                str(max_nb_comp) + ":" + \
                                str(step) + "]" + "]" + \
                                ".npy"
        #"pca_components=" + str(nb_pca_comps) + \
        np.save(output_loc_accuracies, np.array(accuracies))
print(str(nb_comp) + " done")

output_loc_accuracies = "./results/GMMv2/accuracies_covtype="+param["fit"]["cov_type"]+\
                        "_nbinit="+str(param["fit"]["n_init"])+\
                        "["+str(min_nb_comp)+":"+\
                            str(max_nb_comp)+":"+\
                            str(step)+"]"+"]" +\
                            ".npy"
#                        "pca_components="+str(nb_pca_comps)+\

np.save(output_loc_accuracies,np.array(accuracies))