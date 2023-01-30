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

    parser.add_argument("--total-timesteps", type=int, default=500000)
    parser.add_argument("--train-timesteps", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-epsilon", type=int, default=0.05)
    parser.add_argument("--eval-freq", type=int, default=200)
    parser.add_argument("--update-target-timesteps", type=int, default=500)
    parser.add_argument("--start-learning-timestep", type=int, default=10000)
    parser.add_argument("--env-name", type=str, default="CartPole-v1")
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--epsilon-start", type=float, default=1)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-anneal-frac", type=float, default=0.5)
    parser.add_argument("--replay-capacity", type=int, default=10000)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--reward-discount", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--use-wandb", action='store_true')
    parser.add_argument("--wandb-project-name", type=str, default="deep-rl")
    parser.add_argument("--wandb-entity-name", type=str, default="andrew99")
    parser.add_argument("--save-model-freq", type=int, default=-1)
    parser.add_argument("--record-video", action='store_true')
    args = parser.parse_args()
    return args



def get_linear_schedule(start, end, duration, t):
    slope = (end - start) / duration
    return max(start + (slope * t), end)





class DQNAgent():
    def __init__(self, args, run_name):
        self.args = args
        self.env = gym.make(args.env_name)
        self.eval_env = gym.make(args.env_name, render_mode='rgb_array')
        if self.args.record_video:
            video_folder = f"/Users/andrew/dev/deep-rl/runs/{run_name}/videos"
            os.makedirs(video_folder, exist_ok=True)
            self.eval_env = gym.wrappers.RecordVideo(self.eval_env, video_folder=video_folder, episode_trigger=lambda x: x % self.args.eval_episodes == 0)

        observation, info = self.env.reset(seed=args.env_seed)

        self.state_dim = observation.shape[0]
        self.num_actions = self.env.action_space.n

        self.replay_buffer = ReplayBuffer(self.state_dim, args.replay_capacity, args.minibatch_size)
        self.q_network = QNetwork(self.state_dim, self.num_actions)
        self.target_network = QNetwork(self.state_dim, self.num_actions)
        self.update_target_network()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=args.learning_rate)
        self.global_timestep = 0
        self.epsilon = self.args.epsilon_start


    def update_target_network(self):
        self.target_network.model.load_state_dict(self.q_network.model.state_dict())


    def train_episode(self):
        done = False
        observation, info = self.env.reset()
        while not done:
            self.global_timestep += 1
            self.epsilon = get_linear_schedule(self.args.epsilon_start, self.args.epsilon_end, int(self.args.epsilon_anneal_frac * self.args.total_timesteps), self.global_timestep)
            if self.global_timestep < args.start_learning_timestep or random.random() < self.epsilon:
                action = self.env.action_space.sample()
            else:
                q_values = self.q_network(torch.Tensor(observation))
                action = torch.argmax(q_values).item()

            next_observation, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            transition = (observation, action, reward, next_observation, int(terminated or truncated))
            self.replay_buffer.store_transition(transition)

            if self.global_timestep >= args.start_learning_timestep:
                if self.global_timestep % self.args.train_timesteps == 0:
                    minibatch_states, minibatch_actions, minibatch_rewards, minibatch_next_states, minibatch_is_terminal = self.replay_buffer.sample_minibatch()
                    minibatch_is_terminal = minibatch_is_terminal.astype(bool)
                    
                    with torch.no_grad():
                        minibatch_all_q_values = self.target_network(torch.Tensor(minibatch_next_states)).numpy()
                        minibatch_max_q_values = np.amax(minibatch_all_q_values, axis=1)
                        minibatch_targets = minibatch_rewards + (~minibatch_is_terminal * self.args.reward_discount * minibatch_max_q_values)
                        
                    minibatch_all_preds = self.q_network(torch.Tensor(minibatch_states))
                    minibatch_action_preds = minibatch_all_preds[np.arange(len(minibatch_all_preds)), minibatch_actions]
                    
                    loss = F.mse_loss(minibatch_action_preds, torch.Tensor(minibatch_targets), reduction='mean')

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                if self.global_timestep % self.args.update_target_timesteps == 0:
                    self.update_target_network()

            observation = next_observation


    def eval(self):
        returns = []
        mean_td_errors = []
        for ep in range(self.args.eval_episodes):
            rewards = []
            td_errors = []
            done = False
            observation, info = self.eval_env.reset()
            next_q_values = self.q_network(torch.Tensor(observation))

            while not done:
                q_values = next_q_values
                if random.random() < self.args.eval_epsilon:
                    action = self.eval_env.action_space.sample()
                else:
                    q_values = self.q_network(torch.Tensor(observation))
                    action = torch.argmax(q_values).item()

                observation, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                rewards.append(reward)
                if not done:
                    next_q_values = self.q_network(torch.Tensor(observation))
                    target = reward + (self.args.reward_discount * torch.max(next_q_values).item())
                    td_error = abs(target - torch.max(q_values).item())
                    td_errors.append(td_error)
            
            returns.append(sum(rewards))
            mean_td_errors.append(np.mean(td_errors))
        
        return np.mean(returns), np.std(returns), np.mean(mean_td_errors)


    def save_model(self, model_path):
        torch.save(self.q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")


    def cleanup(self):
        self.env.close()
        self.eval_env.close()
                


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 120, bias=True),
            nn.ReLU(),
            nn.Linear(120, 84, bias=True),
            nn.ReLU(),
            nn.Linear(84, output_dim, bias=True),
        )
    
    def forward(self, x):
        return self.model(x)



class ReplayBuffer():
    def __init__(self, state_dim, capacity, minibatch_size):
        self.transition_dim = (2 * state_dim) + 3
        self.state_dim = state_dim
        self.capacity = capacity
        self.current_size = 0
        self.minibatch_size = minibatch_size
        self.next_store_idx = 0
        self.buffer = np.zeros((self.capacity, self.transition_dim))
    
    def store_transition(self, transition):
        (observation, action, reward, next_observation, done) = transition
        transition_array = np.zeros(self.transition_dim)
        transition_array[:self.state_dim] = observation
        transition_array[self.state_dim] = action
        transition_array[self.state_dim + 1] = reward
        transition_array[self.state_dim + 2 : (2 * self.state_dim) + 2] = next_observation
        transition_array[(2 * self.state_dim) + 2] = done
        self.buffer[self.next_store_idx] = transition_array
        self.next_store_idx = (self.next_store_idx + 1) % self.capacity
        self.current_size = min(self.current_size + 1, self.capacity)

    def sample_minibatch(self):
        indices = np.random.choice(self.current_size, self.minibatch_size)
        minibatch = self.buffer[indices]
        minibatch_states = minibatch[:, :self.state_dim]
        minibatch_actions = minibatch[:, self.state_dim]
        minibatch_rewards = minibatch[:, self.state_dim + 1]
        minibatch_next_states = minibatch[:, self.state_dim + 2 : (2 * self.state_dim) + 2]
        minibatch_is_terminal = minibatch[:, (2 * self.state_dim) + 2]
        return minibatch_states, minibatch_actions, minibatch_rewards, minibatch_next_states, minibatch_is_terminal



if __name__ == "__main__":
    args = parse_args()
    exp_name = os.path.basename(__file__).rstrip(".py")
    run_name = f"{args.env_name}__{exp_name}__{args.env_seed}__{int(time.time())}"

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity_name,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
            dir="/Users/andrew/dev/deep-rl"
        )

    writer = SummaryWriter(log_dir=f"/Users/andrew/dev/deep-rl/runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    agent = DQNAgent(args, run_name)
    
    ep = 1
    t0 = time.time()
    while agent.global_timestep < args.total_timesteps:
        agent.train_episode()

        if ep % args.eval_freq == 0:
            t1 = time.time()
            print(f"\n{'=' * 16} EPISODE {ep} {'=' * 16}")
            print(f"Iteration time: {t1 - t0}")
            t0 = t1
            if agent.global_timestep >= args.start_learning_timestep:
                mean_return, std_return, mean_td_error = agent.eval()

                writer.add_scalar("eval/mean_return", mean_return, global_step=agent.global_timestep)
                writer.add_scalar("eval/std_return", std_return, global_step=agent.global_timestep)
                writer.add_scalar("eval/mean_td_error", mean_td_error, global_step=agent.global_timestep)

                print(f"Mean return: {mean_return}\nStd return: {std_return}\nMean TD error: {mean_td_error}\nReplay buffer size: {agent.replay_buffer.current_size}\nEpsilon: {agent.epsilon}\nGlobal timestep: {agent.global_timestep}")
            else:
                print(f"Skipping eval since haven't started training yet.\nReplay buffer size: {agent.replay_buffer.current_size}\nEpsilon: {agent.epsilon}\nGlobal timestep: {agent.global_timestep}")

        if args.save_model_freq > 0 and ep % args.save_model_freq == 0:
            model_path = f"/Users/andrew/dev/deep-rl/runs/{run_name}/{exp_name}__ep{str(ep)}.model"
            agent.save_model(model_path)

        ep += 1

    print(f"\n{'=' * 16} FINAL EVAL {'=' * 16}")
    writer.add_scalar("eval/mean_return", mean_return, global_step=agent.global_timestep)
    writer.add_scalar("eval/std_return", std_return,global_step=agent.global_timestep)
    writer.add_scalar("eval/mean_td_error", mean_td_error,global_step=agent.global_timestep)

    print(f"Mean return: {mean_return}\nStd return: {std_return}\nMean TD error: {mean_td_error}\nReplay buffer size: {agent.replay_buffer.current_size}\nEpsilon: {agent.epsilon}\nGlobal timestep: {agent.global_timestep}")


    agent.cleanup()
    writer.close()

        
    