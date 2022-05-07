import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

N = 7
baseline_scores = (53.52, 50.95, 51.76, 50.51, 53.70,50.00,53.50)
#SVM_scores = (49.39,48.54,48.01,51.88,53.67,51.35,54.64)
#AE_scores = (50.87, 50.87, 49.44, 52.28, 60.00,55.64,56.12)
GMM_scores = (50.80, 49.47, 49.74, 51.46, 62.43,60.27,57.53)
#nn_single_layer_scores = (50.32, 50.61, 50.60, 50.23, 49.34,49.88,49.32)
#svm_scores = (51, 50.27, 50.53, 52.05, 50.15,49.62,50.3)
custom_scores = (52.12,49.64,51.81,51.77,61.72,56.49,55.55)

factor = 3

width = 0.5


ind = np.arange(N)
plt.bar(ind, baseline_scores, width/factor, label='Baseline')
#plt.bar(ind + width/factor, SVM_scores, width/factor, label='One class SVM')
#plt.bar(ind + 2*width/factor, AE_scores, width/factor, label='AE')
plt.bar(ind + width/factor, GMM_scores, width/factor, label='GMM')
plt.bar(ind + 2*width/factor, custom_scores, width/factor, label='custom model')

#plt.bar(ind + 4*width/factor, nn_single_layer_scores, width/factor, label='singe layer NN class')
#plt.bar(ind + 5*width/factor, svm_scores, width/factor, label='SVM class')


#plt.bar(ind + 2*width, women_means, width,
 #   label='One class SVM')

plt.ylabel('pAUC target score')
plt.title('Accuracy per prototype by machine')

plt.xticks(ind + width / 2, ('Fan', 'Pump', 'ToyCar', 'Valve', 'Gearbox','ToyTrain','Slider'))
plt.ylim((0,75))
plt.legend(loc='best')
plt.show()