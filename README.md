# Mountain Car using Deep RL
**[Environment Details](https://github.com/openai/gym/wiki/MountainCar-v0)**

A car is on a one-dimensional track, positioned between two "mountains". The goal is to drive up the mountain on the right; however, the car's engine is not strong enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build up momentum.

**Solved Requirements**

MountainCar-v0 defines "solving" as getting average reward of -110.0 over 100 consecutive trials.

The `target_dqn.h5` model gets an average score of -99.67 in 100 consecutive episodes ans thus solves the Mountain Car environment.

**Files**
1. *target_dqn.h5* : A trained model for predicting Q values for state action pairs.
2. *train.py* : Contains detailed implementation of the DDQN for solving Mountain Car.
3. *q_network.py* : Contains the neural network architechture for predicting Q values.
4. *test.py* : Runs the trained model on the environment for 100 episodes and returns the mean score.

