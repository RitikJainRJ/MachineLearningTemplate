# upper confidence bound algorithm

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# implementing the UCB algo
import math
d = 10
N = 10000
number_of_selections = [0] * d
sum_of_rewards = [0] * d
ads_selected = []
total_rewards = 0
for n in range(0, N):
    max_upper_bound = 0
    ad = 0
    for i in range(0, d):
        if number_of_selections[i] > 0:
            average_reward = sum_of_rewards[i] / number_of_selections[i]
            delta = math.sqrt(3 / 2 * math.log(n + 1) / number_of_selections[i])
            upper_bound = delta + average_reward
        else:
            upper_bound = 1e400
        if max_upper_bound < upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    number_of_selections[ad] += 1
    sum_of_rewards[ad] += reward
    total_rewards += reward
            
        