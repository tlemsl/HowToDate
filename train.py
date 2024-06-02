from sim import DateSimulation
from agent import QLearningAgent
import matplotlib.pyplot as plt
import numpy as np
from game import Game

def play_and_train(env, agent, t_max=100):
    """
    This function should
    - run a full game, actions given by agent's e-greedy policy
    - train agent using agent.update(...) whenever it is possible
    - return total reward and total correct answers
    """
    total_reward = 0.0
    s = 1
    r = 0
    answer=0
    total_answer = 0

    for t in range(t_max):
        a = agent.get_action(s)
        next_s, new_answer, new_r, done = env.step(s, a, answer, r)
        print(f"Stage {s}, Action {a}, and reward {new_r}, answer {new_answer}")
        print(done)

        # train (update) agent for state s
        agent.update(s, a, new_r, next_s)

        s = next_s
        total_reward += new_r
        total_answer += new_answer  # Update total_answer with new_answer
        if done:
            break

    return total_reward, total_answer

if __name__ == "__main__":
    actions = list(range(0, 100))

    agent = QLearningAgent(actions, alpha=0.5, epsilon=0.9, discount=0.99)
    env = DateSimulation(Game)
    rewards = []
    answers = []

    for i in range(1000):
        reward, answer = play_and_train(env, agent)
        rewards.append(reward)
        answers.append(answer)
        agent.epsilon *= 0.99  # Reduce epsilon to decrease exploration over time
        print(f"Episode {i+1}: Reward = {reward}, Correct Answers = {answer}")

    plt.title('Epsilon = {:e}, Mean Reward = {:.1f}'.format(agent.epsilon, np.mean(rewards[-10:])))
    plt.plot(rewards, label='Total Reward')
    #plt.plot(answers, label='Total Correct Answers')
    plt.legend()
    plt.show()