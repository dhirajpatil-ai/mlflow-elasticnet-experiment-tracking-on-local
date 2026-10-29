import numpy as np
from model import train

alpha_values = np.linspace(0.1, 0.5, 5)
l1_ratio_values = np.linspace(0.1, 0.5, 5)

for alpha in alpha_values:
    for l1_ratio in l1_ratio_values:
        train(alpha, l1_ratio)