
# A O(sqrt(n)) java program that prints
# all divisors in sorted order
import math

# Method to print the divisors
import matplotlib.pyplot

lrs = []
lr = 1
for step in range(200):
    lrs.append(lr*10** 0.5* min(step ** -0.5, step * 10 ** -1.5))

matplotlib.pyplot.plot(lrs)