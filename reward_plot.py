
import matplotlib.pyplot as plt
import utill
import numpy as np
rewards_file = "data/rewards2.csv"
try:
    rewards = utill.load_rewards(rewards_file)
except FileNotFoundError:
    rewards = [-35, -23, -44, -39, -40, -14, -30, -6]
plt.title('mean reward = {:.1f}'.format(np.mean(rewards[-10:])))
plt.plot(rewards)
plt.show()  # Clear the current figure to update the plot in the next iteration