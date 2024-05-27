from sim import DateSimulation
from agent import QLearningAgent


def play_and_train(env, agent, t_max=10**4):
    """
    This function should
    - run a full game, actions given by agent's e-greedy policy
    - train agent using agent.update(...) whenever it is possible
    - return total reward
    """
    total_reward = 0.0
    s, _ = env.reset()

    for t in range(t_max):
        # get agent to pick action given state s.
        a = agent.get_action(s)

        next_s, r, done, _, _ = env.step(a)

        # train (update) agent for state s
        agent.update(s, a,r,next_s)

        s = next_s
        total_reward += r
        if done:
            break

    return total_reward

if __name__ == "__main__":
    actions = []
    for i in range(100):
        for j in range(100):
            for t in ["d", "c"]:
                actions.append((i,j,t))
    
    env = DateSimulation(resolution="1024x576")
    agent = QLearningAgent(alpha=0.5, epsilon=0.2, discount=0.99,
                           get_legal_actions=actions)
    play_and_train(env=env, agent=agent)