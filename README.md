# Deep RL
My implementations of popular deep reinforcement algorithms. Each algorithm is implemented in a single file for readability and ease of understanding.

## Algorithms Implemented
| Algorithm                                                                                          | Action Space | Implementation                                                                           |
|----------------------------------------------------------------------------------------------------|--------------|------------------------------------------------------------------------------------------|
| [Deep Q-learning (DQN)](https://arxiv.org/pdf/1312.5602.pdf)                                       | Discrete     | [`dqn.py`](https://github.com/andrewsingh/deep-rl/blob/main/algorithms/dqn.py)             |
| [REINFORCE (with baseline)](https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf) | Discrete     | [`reinforce.py`](https://github.com/andrewsingh/deep-rl/blob/main/algorithms/reinforce.py) |
| [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/pdf/1509.02971.pdf)                  | Continuous   | [`ddpg.py`](https://github.com/andrewsingh/deep-rl/blob/main/algorithms/ddpg.py)           |
| [Twin Delayed Deep Deterministic Policy Gradient (TD3)](https://arxiv.org/pdf/1802.09477.pdf)      | Continuous   | [`td3.py`](https://github.com/andrewsingh/deep-rl/blob/main/algorithms/td3.py)             |


## Algorithms in Progress
- [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf)

## Experiments
### DQN vs. REINFORCE with / without baseline on Cart Pole
![dqn-vs-reinforce-cartpole-resized](https://user-images.githubusercontent.com/20130365/216112242-36c6be1e-ca60-45cc-802f-b9640b89c885.png)


### DDPG vs. TD3 on Half Cheetah
![ddpg-vs-td3-halfcheetah-resized](https://user-images.githubusercontent.com/20130365/216112056-788ae798-1f3e-4462-9ac2-288dd756a6bf.png)
