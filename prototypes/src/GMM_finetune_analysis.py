import numpy as np
import matplotlib.pyplot as plt

######################################################
# constants and helper functions
######################################################


baseline_accs=[
    [67.49, 60.09, 53.64, 53.52],
    [71.26, 56.05, 58.52, 50.95],
    [67.99, 58.40, 53.08, 51.76],
    [54.59, 52.89, 50.70, 50.51],
    [62.59, 70.94, 51.90, 53.70],
    [71.74, 54.26, 59.79, 50.00],
    [78.20, 52.89, 59.30, 53.50]
]

machines=["fan","pump","ToyCar","valve","gearbox","ToyTrain","slider"] # same ordder as the files are stored

metrics=["AUC_source","pAUC_source","AUC_target","pAUC_target"]

colors=["r","g","b","magenta"]



def process_loaded_file(accuracies_machine_loaded,offset_till_metric):
    accuracies_machine_processed = []

    for metric_offset in range(len(metrics)):
        accuracies_machine_metric = []
        for sublist in accuracies_machine_loaded:
            accuracies_machine_metric.append(sublist[offset_till_metric + metric_offset])
        accuracies_machine_processed.append(accuracies_machine_metric)

    return accuracies_machine_processed



def merge_accuracy_files(source_loc1, source_loc2, target_loc):
    accuracies1_loaded = np.ndarray.tolist(np.load(source_loc1))
    accuracies2_loaded = np.ndarray.tolist(np.load(source_loc2))
    result_accs = accuracies1_loaded
    for i in range(7):
        for el in accuracies2_loaded[i]:
            result_accs[i].append(el)
    print(accuracies1_loaded)
    result_accs_to_save=np.array(accuracies1_loaded)
    np.save(target_loc,result_accs_to_save)



######################################################
# Actual generation of plots: quick and dirty
######################################################


source_loc1="./results/GMM/accuracies_covtype=full_nbinit=10[30:85:5]].npy"
source_loc2="./results/GMM/accuracies_covtype=full_nbinit=10[90:100:5]].npy"
target_loc="./results/GMM/accuracies_covtype=full_nbinit=10[30:100:5]].npy"
merge_accuracy_files(source_loc1, source_loc2, target_loc)



source_loc="./results/GMM/accuracies_covtype=full_nbinit=10[1:25:1]].npy"
source_loc_extra="./results/GMM/accuracies_covtype=full_nbinit=10[30:100:5]].npy"



accuracies_loaded = np.load(source_loc)
accuracies_loaded_extra = np.load(source_loc_extra)

offset_till_metric = 1

for machine_offset in range(len(machines)):
    machine = machines[machine_offset]
    accuracies_machine_loaded = accuracies_loaded[machine_offset]
    accuracies_machine_loaded_extra = accuracies_loaded_extra[machine_offset]

    accuracies_machine_processed = process_loaded_file(accuracies_machine_loaded, offset_till_metric)
    accuracies_machine_processed_extra = process_loaded_file(accuracies_machine_loaded_extra, offset_till_metric)

    print(accuracies_machine_processed)
    output_location = "./results/GMM/fine_tune_plots/accuracy GMM " + machine

    x_axis_old = [k + 1 for k in range(len(accuracies_machine_processed[0]))]

    accuracies_together = accuracies_machine_processed

    for i in range(len(accuracies_machine_processed_extra)):
        accuracies_together[i] = accuracies_together[i] + accuracies_machine_processed_extra[i]

    start=25
    step=5
    nb_new=15

    x_axis_new=[start+step+k*step for k in range(nb_new)]

    x_axis_final = x_axis_old+x_axis_new
    xticks_old=[str(1)]
    for i in range(1,25):
        if (i%step)==step-1:
            xticks_old.append(str(i+1))
        else:
            xticks_old.append('')
    xticks_final=xticks_old+[str(30),str(35),str(40),str(45),str(50),str(55),str(60),str(65)
                             ,str(70),str(75),str(80),str(85),str(90),str(95),str(100)]

    fig, ax = plt.subplots()

    machine=machines[machine_offset]
    for i in range(len(accuracies_together)):
        plt.plot(x_axis_final,accuracies_together[i], label = metrics[i],color=colors[i])
    basline_accs_machine = baseline_accs[machine_offset]
    for j in range(len(basline_accs_machine)):
        score_to_plot = [basline_accs_machine[j] / 100 for k in range(len(x_axis_final))]
        plt.plot(x_axis_final, score_to_plot, linestyle="dotted", color=colors[j])
    ax.set_xticks(x_axis_final)
    ax.set_xticklabels(xticks_final)
    plt.title("accuracy GMM " + machine)
    plt.xlabel("number of components GMM")
    plt.ylabel("accuracy")
    plt.legend()

    plt.savefig(output_location)
    plt.close()