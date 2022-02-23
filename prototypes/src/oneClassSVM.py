import os
import numpy as np
from sklearn.svm import OneClassSVM
import ADAST
import pandas as pd
from ASD_ASTplus.Gilles.utilities import common
import csv

result_dir="./results/oneClassSVM/"
f = open(result_dir+"AUC_scores_oneClassSVM.txt", 'w')
f_csv = open(result_dir+"AUC_scores_oneClassSVM.csv", 'w')

csv_writer = csv.writer(f_csv)
header=["machine","AUC_source","pAUC_source","AUC_target","pAUC_target"]
csv_writer.writerow(header)

#combine individual tensors stored in .pt to one pandas dataframe and save as pkl file => not enough RAM
skip_count=0
embedding_base_directory="../dev_data_embeddings/"
for machine in os.listdir(embedding_base_directory):
    if os.path.isdir(embedding_base_directory+"/"+machine):
        print(machine)
        nu=0.01
        source_test_FPR_TPRs=[]
        target_test_FPR_TPRs=[]
        while(nu<=1):
            print("nu="+str(nu))

            machine_dir=embedding_base_directory + machine
            train_pickle_location=machine_dir+"/train/"+"dataframe.pkl"
            lables_train,X= ADAST.generate_lables_and_pd_dataframe(machine_dir + "/train/")
            X=pd.read_pickle(train_pickle_location)


            one_class_SVM_model = OneClassSVM(nu=nu)
            one_class_SVM_model.fit(X)
            train_prediction=one_class_SVM_model.predict(X)

            train_prediction=one_class_SVM_model.predict(X)
            predicted_anomaly=(train_prediction==-1).sum()
            true_pos, true_neg, false_pos, false_neg = common.generate_confusion_matrix(train_prediction, lables_train)
            print("TRAIN")
            print("True pos: " + str(true_pos) +", false pos: "+str(false_pos))
            print("False neg: " + str(false_neg) +", True neg: "+str(true_neg))



            labels_source_test,X_source_test=ADAST.generate_lables_and_pd_dataframe(machine_dir + "/source_test")
            source_prediction=one_class_SVM_model.predict(X_source_test)
            true_pos, true_neg, false_pos, false_neg = common.generate_confusion_matrix(source_prediction, labels_source_test)
            print("SOURCE TEST")
            print("True pos: " + str(true_pos) +", false pos: "+str(false_pos))
            print("False neg: " + str(false_neg) +", True neg: "+str(true_neg))
            FPR=false_pos/(false_pos+true_neg)
            TPR=true_pos/(true_pos+false_neg)
            source_test_FPR_TPRs.append((FPR,TPR))


            labels_target_test,X_target_test=ADAST.generate_lables_and_pd_dataframe(machine_dir + "/target_test")
            target_prediction=one_class_SVM_model.predict(X_target_test)
            true_pos, true_neg, false_pos, false_neg = common.generate_confusion_matrix(target_prediction, labels_target_test)
            print("Target TEST")
            print("True pos: " + str(true_pos) +", false pos: "+str(false_pos))
            print("False neg: " + str(false_neg) +", True neg: "+str(true_neg))
            FPR=false_pos/(false_pos+true_neg)
            TPR=true_pos/(true_pos+false_neg)
            target_test_FPR_TPRs.append((FPR,TPR))

            # skip_count+=1

            nu+=0.01
    print(source_test_FPR_TPRs)
    auc_approx_source_test= common.calculate_pAUC_approximation(source_test_FPR_TPRs, p=1)
    print("AUC_approx_source_test: "+str(auc_approx_source_test))
    pauc_approx_source_test= common.calculate_pAUC_approximation(source_test_FPR_TPRs, p=0.1)
    print("pAUC_approx_source_test: "+str(pauc_approx_source_test))

    print()
    print(target_test_FPR_TPRs)
    auc_approx_target_test= common.calculate_pAUC_approximation(target_test_FPR_TPRs, p=1)
    print("AUC_approx_target_test: "+str(auc_approx_target_test))
    pauc_approx_target_test= common.calculate_pAUC_approximation(target_test_FPR_TPRs, p=0.1)
    print("pAUC_approx_target_test: "+str(pauc_approx_target_test))
    f.write(str(machine)+": AUC source test="+str(auc_approx_source_test)+", pAUC source test="+str(pauc_approx_source_test) +
            "\n \t AUC target test="+str(auc_approx_target_test)+", pAUC target test="+str(pauc_approx_target_test)+
            "\n")

    csv_row=[machine,auc_approx_source_test,pauc_approx_source_test,auc_approx_target_test,pauc_approx_target_test]
    csv_writer.writerow(csv_row)

    AUC_curve_output_location=result_dir+"AUC_target" +str(machine)
    common.generate_ROC_curve(target_test_FPR_TPRs, AUC_curve_output_location)

    AUC_curve_output_location=result_dir+"AUC_source" +str(machine)
    common.generate_ROC_curve(source_test_FPR_TPRs, AUC_curve_output_location)

    FPR_TPR_curve_output_location_source=result_dir+"/numpy_vectors/FPR_TPRs_source_test_" +str(machine)
    np.save(FPR_TPR_curve_output_location_source,source_test_FPR_TPRs)

    FPR_TPR_curve_output_location_target=result_dir+"/numpy_vectors/FPR_TPRs_target_test_" +str(machine)
    np.save(FPR_TPR_curve_output_location_target,target_test_FPR_TPRs)
    print(machine + " done")



f.close()
f_csv.close()


