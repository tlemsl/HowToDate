from sim import DateSimulation
from agent import QLearningAgent
import pyautogui
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
    model_path = "data/test.pkl"
    actions = []
    for i in range(20):
        for j in range(15):
            for t in ["c"]:     
                actions.append((i,j,t))
    place, start_img= utill.reset_position() 
    start_arr=np.array(start_img)
    start_tuple=utill.numpy2tuple(start_arr)
    env = DateSimulation(place, start_tuple)
    
    agent = QLearningAgent(model_path, actions, alpha=0.5, epsilon=0.7, discount=0.99)
    agent.load_qvalues()
    #play_and_train(env=env, agent=agent)
    #env.reset()
    rewards = []
    for i in range(1000):
        rewards.append(play_and_train(env, agent))
        agent.epsilon *= 0.9 #값 수정함a
        agent.save_qvalues()
        print(rewards[-1])
        if True:
            plt.title('eps = {:e}, mean reward = {:.1f}'.format(agent.epsilon, np.mean(rewards[-10:])))
            plt.plot(rewards)
            plt.show()
