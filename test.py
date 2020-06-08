from keras.models import load_model
import gym
import numpy as np

ENV_NAME = 'MountainCar-v0'
model = load_model('target_dqn.h5')
EPISODES = 100


def run_episodes(env):
    observation_space = env.observation_space.shape[0]
    avg_reward = 0
    for ep in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        game_rew = 0
        while True:
            env.render()
            step += 1
            q_values = model.predict(state)
            action = np.argmax(q_values[0])
            next_state, rew, done, info = env.step(action)
            rew = rew if not done else -rew
            next_state = np.reshape(next_state, [1, observation_space])
            state = next_state
            game_rew += rew
            if done:
                print("Epoch :" + str(ep) + " Score :" + str(step))
                avg_reward += game_rew
                break

    avg_reward /= EPISODES
    print("\n\nAverage Reward  in " + str(EPISODES) + " games : " +
          str(avg_reward))


if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    run_episodes(env)
