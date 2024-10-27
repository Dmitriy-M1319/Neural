import math
import numpy as np
import matplotlib.pyplot as plt


# Отрисовка графиков по вариантам
def function(x, a):
    return (math.exp(a*x) - math.exp(-a*x)) / (math.exp(a*x) + math.exp(-a*x))

alpha = 0.7

x_arr = np.linspace(-1, 1, 101)
y_arr = [function(x, alpha) for x in x_arr]

plt.plot(x_arr, y_arr)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.show()
