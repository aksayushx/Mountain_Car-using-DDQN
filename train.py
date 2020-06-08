import gym
import numpy as np
import random
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
from q_network import q_network

ENV_NAME = 'MountainCar-v0'

GAMMA = 0.99
EPS_INITIAL = 0.95
EPS_FINAL = 0.01
EPS_DECAY = 0.95

EPISODES = 750
BATCH_SIZE = 32
LR = 0.001
MEMORY_SIZE = 1000000


class Dqn:
    def __init__(self, action_space):
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.eps = EPS_INITIAL

    def add_experience(self, obs, act, rew, next_obs, done):
        self.memory.append((obs, act, rew, next_obs, done))

    def action(self, state, dqn):
        if np.random.uniform(0, 1) < self.eps:
            return random.randrange(self.action_space)

        q_values = dqn.predict(state)
        return np.argmax(q_values[0])

    def exp_replay(self, dqn, target_dqn):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)

        for obs, act, rew, next_obs, done in batch:
            q_update = rew
            if not done:
                next_act = np.argmax(dqn.predict(next_obs)[0])
                action_value = target_dqn.predict(next_obs)
                q_update = rew + GAMMA * action_value[0][next_act]
            q_values = dqn.predict(obs)
            q_values[0][act] = q_update
            dqn.fit(obs, q_values, verbose=0)

        self.eps *= EPS_DECAY
        self.eps = max(self.eps, EPS_FINAL)


def agent():
    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    Q = Dqn(action_space)
    target_dqn = q_network(observation_space, action_space)
    dqn = q_network(observation_space, action_space, 0.001)

    for ep in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        game_rew = 0

        while True:
            #uncomment this to see the car performance during training
            #env.render()
            step += 1
            action = Q.action(state, dqn)
            next_state, rew, done, info = env.step(action)
            if done and step < 200:
                rew += 10000 // step
            next_state = np.reshape(next_state, [1, observation_space])
            Q.add_experience(state, action, rew, next_state, done)
            state = next_state
            game_rew += rew
            if ep % 100 == 0 and done:
                dqn.save('dqn_' + str(ep) + '.h5')
            if done:
                target_dqn.set_weights(dqn.get_weights())
                print("Episode :" + str(ep) + " Score :" + str(-1 * step))
                break
            Q.exp_replay(dqn, target_dqn)

    target_dqn.save('target_dqn.h5')


if __name__ == '__main__':
    agent()
