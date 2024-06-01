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
        while r==0:       
            a = agent.get_action(s)
            next_s, r, done= env.step(s, a)
            print(f"Action {a}, and reward {r}")

        # train (update) agent for state s
        agent.update(s, a,r,next_s)

        s = next_s
        total_reward += r
        if done:
            break

    return total_reward

if __name__ == "__main__":
    actions = []
    for i in range(20):
        for j  in range(15):          
            for t in ["c"]:     
                actions.append((i,j,t))
    place, start_img= utill.reset_position() 
    #start_img.crop((250,100,450,300)).show()
    start_arr=np.array(start_img)
    start_tuple=utill.numpy2tuple(start_arr)
    env = DateSimulation(place, start_tuple)
    actions=[(519, 350,'c'), (361, 351, 'c'), (516, 374,'c'), (160, 316, 'c'), (333, 124, 'c'), (133, 75,'c'), (135, 26, 'c'), (344, 200, 'c'), (302, 261, 'c'), (396, 269, 'c'), (516, 316, 'c')]  #수정

    
    #actions.append(( 16, 14, 'c'))
    #print(actions)

    agent = QLearningAgent(actions, alpha=0.5, epsilon=0.7, discount=0.99)

    rewards = []
    for i in range(1000):
        rewards.append(play_and_train(env, agent))
        agent.epsilon *= 0.95 #값 수정함
        print(rewards[-1])
    plt.title('eps = {:e}, mean reward = {:.1f}'.format(agent.epsilon, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.show()
