import gym
from agent import Agent
from tqdm import tqdm
import numpy as np
env = gym.make('LunarLander-v2')
env.reset ()
new_state, reward, done,_ = env.step (0)
observation_space = new_state.shape
print (observation_space)
agent = Agent (observation_space,env.action_space.n)
N_EPISODES = 30000
SHOW_EVERY = 50
SAVE_EVERY = 15000
MIN_EPSILON = 0.001
EPSILON_DECAY = 0.99975
epsilon = 1


for episode in tqdm (range (N_EPISODES),ascii = True,unit = "episode"):
    episode_reward = 0
    step = 1
    current_state = env.reset()
    done = False
    while not done:
        if np.random.random () > epsilon:
            action = np.argmax(agent.get_qs(current_state))
        else :
            action = np.random.randint(0,env.action_space.n)
        new_state, reward, done,_ = env.step (action)
        episode_reward += reward
        if not episode % SHOW_EVERY:
            env.render ()
        if not (episode % SAVE_EVERY) and episode:
            print ('saved')
            agent.save_model ()
        agent.update_replay_memory((current_state,action,reward,new_state,done))
        agent.train(done,step)
        corrent_state = new_state
        step+=1
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)
env.close()
