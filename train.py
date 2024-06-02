from sim import DateSimulation
from agent import QLearningAgent
import time
import matplotlib.pyplot as plt
import utill
import numpy as np

def play_and_train(env, agent, t_max=1000):
    """
    This function should
    - run a full game, actions given by agent's e-greedy policy
    - train agent using agent.update(...) whenever it is possible
    - return total reward
    """
    total_reward = 0.0
    s=env.reset()      
    for t in range(t_max):
        
        # get agent to pick action given state s.
        r=0
        # for i in range(30):
        #     pyautogui.press('enter')
        #     time.sleep(0.2)
        next_s = s      
        # while r==0:
        start_time = time.time()
        a = agent.get_action(s)
        print(f"Do Action: {a}")

        next_s, r, done= env.step(s, a)
        print(f"Reward: {r} Feedback time {time.time() - start_time}")

        # train (update) agent for state s
        agent.update(s, a,r,next_s)

        s = next_s
        total_reward += r
        if done:
            break

    return total_reward

if __name__ == "__main__":
    model_path = "data/train"
    rewards_file = "data/rewards2.csv"
    try:
        rewards = utill.load_rewards(rewards_file)
    except FileNotFoundError:
        # rewards = [-35, -23, -44, -39, -40, -14, -30, -6]
        rewards = []
    
    place, start_img= utill.reset_position() 
    start_arr=np.array(start_img)
    start_tuple=utill.numpy2tuple(start_arr)
    env = DateSimulation(place, start_tuple)
    
    actions=[(376, 351, 'c'), (110, 316, 'c'), (333, 124, 'c'), (344, 200, 'c'), (302, 261, 'c'), (396, 269, 'c')]

    pre_trained_epoch = len(rewards)
    agent = QLearningAgent(actions, alpha=0.5, epsilon=0.7 * 0.9 * pre_trained_epoch, discount=0.99)

    if pre_trained_epoch == 0:
        agent = QLearningAgent(actions, alpha=0.5, epsilon=0.7 * 0.9, discount=0.99)
    agent.load_qvalues(f"{model_path}12.pkl")
    
    
    play_times = []
    for i in range(pre_trained_epoch, 1000):
        start_time = time.time()
        rewards.append(play_and_train(env, agent))
        play_times.append(time.time() - start_time)
        agent.epsilon *= 0.9 #값 수정함a
        print(f"Epoch {i}, Rewards {rewards[-1]}, Play time {play_times[-1]}")
        utill.save_rewards(rewards, rewards_file)
        if i % 3 == 0 or rewards[-1] > 0:
            agent.save_qvalues(f"{model_path}{i}.pkl")
        
        # if True:
        #     plt.title('eps = {:e}, mean reward = {:.1f}'.format(agent.epsilon, np.mean(rewards[-10:])))
        #     plt.plot(rewards)
        #     plt.draw()
        #     plt.pause(0.001)  # Add a short pause to allow the plot to update
        #     plt.clf()  # Clear the current figure to update the plot in the next iteration
