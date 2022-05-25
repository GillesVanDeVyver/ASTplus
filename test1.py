

# Method to print the divisors
import matplotlib.pyplot as plt
import numpy as np

lrs = []
lr = 1
p = 10
for step in range(1,200):
    lrs.append(lr*p** 0.5* min(step ** -0.5, step * p ** -1.5))
print(lrs)

lrs2 = []
lr = 1
lr_curr = lr
g = 0.5
ms = [2,3,4,5]
for step in range(1,200):
    if step in ms:
        lr_curr = lr_curr*g
    lrs2.append(lr_curr)
print(lrs2)


plt.plot(lrs)
plt.plot(lrs2)
plt.show()


def p(t):
    """Basic rectangular pulse"""
    return 1 * (abs(t) < 0.5)


functions = [p]

t = np.linspace(-2, 2, 1000)

plt.figure()
for i, function in enumerate(functions, start=1):
    plt.subplot(1, 1, i)
    plt.plot(t, function(t), '-b')
    plt.ylim((-1.1, 1.1))
    plt.title(function.__doc__)
plt.tight_layout()
plt.show()