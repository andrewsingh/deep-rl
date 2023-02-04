import gymnasium as gym
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import wandb
import os
import time
import pdb

from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser()

    # Experiment settings
    parser.add_argument("--env-name", type=str, default="ALE/Breakout-v5")
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=10000000)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--eval-num-episodes", type=int, default=10)

    # Network hyperparams
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--update-freq", type=int, default=4)
    parser.add_argument("--update-target-freq", type=int, default=1000)

    # Other hyperparams
    parser.add_argument("--epsilon-start", type=float, default=1)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-anneal-frac", type=float, default=0.1)
    parser.add_argument("--eval-epsilon", type=int, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=1)
    parser.add_argument("--replay-capacity", type=int, default=1000000)
    parser.add_argument("--start-learning-timestep", type=int, default=50000)
    
    # Logging
    parser.add_argument("--cpu", action='store_true')
    parser.add_argument("--use-tensorboard", action='store_true')
    parser.add_argument("--use-wandb", action='store_true')
    parser.add_argument("--wandb-project-name", type=str, default="deep-rl")
    parser.add_argument("--wandb-entity-name", type=str, default="andrew99")
    parser.add_argument("--record-video", action='store_true')
    parser.add_argument("--record-video-eval-freq", type=int, default=5)

    args = parser.parse_args()
    return args


# DQN algorithm
def dqn(args, env, eval_env, writer=None):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Device: {device}")

    observation, info = env.reset(seed=args.env_seed)
    state_shape = observation.shape
    num_actions = env.action_space.n
    print(f"Num actions: {num_actions}")

    # Initialize Q networks and replay buffer
    q_network = QNetwork(num_actions).to(device)
    target_network = QNetwork(num_actions).to(device)
    target_network.load_state_dict(q_network.state_dict())

    replay_buffer = ReplayBuffer(state_shape, args.replay_capacity, args.minibatch_size)

    optimizer = torch.optim.Adam(q_network.parameters(), lr=args.lr)
    global_timestep = 0
    epsilon = args.epsilon_start

    global_timestep = 0
    ep = 0
    t0 = time.time()
    while global_timestep < args.total_timesteps:
        ep += 1
        done = False
        observation, info = env.reset()
        
        # Run a single episode
        while not done:
            global_timestep += 1

            epsilon = get_linear_schedule(args.epsilon_start, args.epsilon_end, int(args.epsilon_anneal_frac * args.total_timesteps), global_timestep)

            # Select action
            if global_timestep < args.start_learning_timestep or random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = q_network(torch.Tensor(np.array(observation)).unsqueeze(0).to(device))
                action = torch.argmax(q_values).item()

            # Execute action in environment and store transition in replay buffer
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            transition = (np.array(observation), action, reward, np.array(next_observation), int(terminated or truncated))
            replay_buffer.store_transition(transition)

            if global_timestep >= args.start_learning_timestep:
                if global_timestep % args.update_freq == 0:
                    # Sample random minibatch of transitions from replay buffer
                    minibatch_states, minibatch_actions, minibatch_rewards, minibatch_next_states, minibatch_is_terminal = replay_buffer.sample_minibatch()
                    minibatch_is_terminal = minibatch_is_terminal.astype(bool)
                    
                    # Calculate Q network targets
                    with torch.no_grad():
                        minibatch_all_q_values = target_network(torch.Tensor(minibatch_next_states).to(device)).cpu().numpy()
                        minibatch_max_q_values = np.amax(minibatch_all_q_values, axis=1)
                        minibatch_targets = minibatch_rewards + (~minibatch_is_terminal * args.gamma * minibatch_max_q_values)
                        minibatch_targets_t = torch.Tensor(minibatch_targets).to(device)

                    # Get Q network predictions
                    minibatch_all_preds = q_network(torch.Tensor(minibatch_states).to(device))
                    minibatch_action_preds = minibatch_all_preds[np.arange(len(minibatch_all_preds)), minibatch_actions]
                    
                    # Calculate Q network loss and perform gradient descent step
                    loss = F.mse_loss(minibatch_action_preds, minibatch_targets_t, reduction='mean')
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Delayed target network updates
                if global_timestep % args.update_target_freq == 0:
                    soft_update_target_network(target_network, q_network, args.tau)

            observation = next_observation

            # Evaluation
            if global_timestep % args.eval_freq == 0:
                t1 = time.time()
                print(f"\n{'=' * 16} TIMESTEP {global_timestep} {'=' * 16}")
                print(f"Iteration time: {t1 - t0}\nCurrent episode: {ep}")
                t0 = t1

                if global_timestep >= args.start_learning_timestep:
                    mean_return, std_return = evaluate(eval_env, q_network, args.eval_num_episodes, device=device)

                    if writer != None:
                        writer.add_scalar("eval/mean_return", mean_return, global_step=global_timestep)
                        writer.add_scalar("eval/std_return", std_return, global_step=global_timestep)

                    print(f"Mean return: {mean_return}\nStd return: {std_return}\nReplay buffer size: {replay_buffer.current_size}\nTrain epsilon: {epsilon}")
                else:
                    print(f"Skipping eval - still filling the replay buffer.\nReplay buffer size: {replay_buffer.current_size}\nTrain epsilon: {epsilon}")


def evaluate(eval_env, q_network, num_episodes, device="cpu"):
    ep_returns = []
    for ep in range(num_episodes):
        ep_return = 0
        done = False
        observation, info = eval_env.reset()

        while not done:
            if random.random() < args.eval_epsilon:
                action = eval_env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_network(torch.Tensor(np.array(observation)).unsqueeze(0).to(device))
                    action = torch.argmax(q_values).item()
            
            observation, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            ep_return += reward
        
        ep_returns.append(ep_return)
    
    return np.mean(ep_returns), np.std(ep_returns)


def soft_update_target_network(target_network, local_network, tau):
    for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def get_linear_schedule(start, end, duration, t):
    slope = (end - start) / duration
    return max(start + (slope * t), end)



class QNetwork(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        return self.model(x / 255.0)



class ReplayBuffer():
    def __init__(self, state_shape, capacity, minibatch_size):
        self.transition_dim = 3
        self.capacity = capacity
        self.current_size = 0
        self.minibatch_size = minibatch_size
        self.next_store_idx = 0
        self.buffer = np.zeros((self.capacity, self.transition_dim))
        self.state_buffer = np.zeros((self.capacity,) + state_shape, dtype=np.uint8)
        self.next_state_buffer = np.zeros((self.capacity,) + state_shape, dtype=np.uint8)
    
    def store_transition(self, transition):
        (observation, action, reward, next_observation, done) = transition
        transition_array = np.zeros(self.transition_dim)
        transition_array[0] = action
        transition_array[1] = reward
        transition_array[2] = done
        self.buffer[self.next_store_idx] = transition_array
        self.state_buffer[self.next_store_idx] = observation
        self.next_state_buffer[self.next_store_idx] = next_observation
        self.next_store_idx = (self.next_store_idx + 1) % self.capacity
        self.current_size = min(self.current_size + 1, self.capacity)

    def sample_minibatch(self):
        indices = np.random.choice(self.current_size, self.minibatch_size)
        minibatch = self.buffer[indices]
        minibatch_actions = minibatch[:, 0]
        minibatch_rewards = minibatch[:, 1]
        minibatch_is_terminal = minibatch[:, 2]
        minibatch_states = self.state_buffer[indices]
        minibatch_next_states = self.next_state_buffer[indices]
        return minibatch_states, minibatch_actions, minibatch_rewards, minibatch_next_states, minibatch_is_terminal



if __name__ == "__main__":
    args = parse_args()
    root_dir = os.path.dirname(os.getcwd())
    exp_name = os.path.basename(__file__).rstrip(".py")
    run_name = f"{args.env_name}__{exp_name}__{args.env_seed}__{int(time.time())}"
    
    # Logging setup
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity_name,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
            dir=root_dir
        )

    writer = None
    if args.use_tensorboard or args.use_wandb:
        writer = SummaryWriter(log_dir=f"{root_dir}/runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    
    # Environment setup
    env = gym.make(args.env_name, frameskip=1)
    env = gym.wrappers.AtariPreprocessing(env)
    env = gym.wrappers.FrameStack(env, num_stack=4)

    eval_env = gym.make(args.env_name, frameskip=1, render_mode='rgb_array')
    eval_env = gym.wrappers.AtariPreprocessing(eval_env)
    eval_env = gym.wrappers.FrameStack(eval_env, num_stack=4)

    if args.record_video:
        video_folder = f"{root_dir}/runs/{run_name}/videos"
        os.makedirs(video_folder, exist_ok=True)
        eval_env = gym.wrappers.RecordVideo(eval_env, video_folder=video_folder, episode_trigger=lambda x: x % (args.eval_num_episodes * args.record_video_eval_freq) == 0)

    
    t_start = time.time()

    # Run DQN algorithm
    dqn(args, env, eval_env, writer=writer)

    t_end = time.time()
    print(f"Total time elapsed: {t_end - t_start}")

    # Cleanup
    env.close()
    eval_env.close()
    writer.close()

        
    