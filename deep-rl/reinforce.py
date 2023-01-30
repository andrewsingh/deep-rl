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
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-freq", type=int, default=200)
    parser.add_argument("--env-name", type=str, default="CartPole-v1")
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--reward-discount", type=float, default=0.99)
    parser.add_argument("--policy-lr", type=float, default=5e-4)
    parser.add_argument("--state-value-lr", type=float, default=5e-4)
    parser.add_argument("--policy-hidden-dim", type=float, default=20)
    parser.add_argument("--state-value-hidden-dim", type=float, default=20)
    parser.add_argument("--use-tensorboard", action='store_true')
    parser.add_argument("--use-wandb", action='store_true')
    parser.add_argument("--wandb-project-name", type=str, default="deep-rl")
    parser.add_argument("--wandb-entity-name", type=str, default="andrew99")
    parser.add_argument("--save-model-freq", type=int, default=-1)
    parser.add_argument("--record-video", action='store_true')
    parser.add_argument("--use-baseline", action='store_true')
    args = parser.parse_args()
    return args



class ReinforceAgent():
    def __init__(self, args, run_name):
        self.args = args

        self.gamma = self.args.reward_discount

        self.env = gym.make(args.env_name)
        self.eval_env = gym.make(args.env_name, render_mode='rgb_array')
        if self.args.record_video:
            video_folder = f"/Users/andrew/dev/deep-rl/runs/{run_name}/videos"
            os.makedirs(video_folder, exist_ok=True)
            self.eval_env = gym.wrappers.RecordVideo(self.eval_env, video_folder=video_folder, episode_trigger=lambda x: x % self.args.eval_episodes == 0)

        observation, info = self.env.reset(seed=args.env_seed)

        self.state_dim = observation.shape[0]
        self.action_dim = self.env.action_space.n
        
        self.policy_network = ReinforceNetwork(self.state_dim, self.action_dim, self.args.policy_hidden_dim)
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=args.policy_lr)

        if self.args.use_baseline:
            self.state_value_network = ReinforceNetwork(self.state_dim, 1, self.args.state_value_hidden_dim)
            self.state_value_optimizer = torch.optim.Adam(self.state_value_network.parameters(), lr=args.state_value_lr)

        self.global_timestep = 0

    
    def generate_episode(self):
        states = []
        selected_action_log_probs = []
        rewards = []
        done = False
        observation, info = self.env.reset()
        while not done:
            self.global_timestep += 1
            states.append(observation)
            action_logits = self.policy_network(torch.Tensor(observation))
            action_log_probs = F.log_softmax(action_logits, dim=0)
            action_probs = F.softmax(action_logits, dim=0).detach().numpy()
            action = np.random.choice(self.action_dim, p=action_probs)
            selected_action_log_probs.append(action_log_probs[action])

            observation, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            rewards.append(reward)

        return np.array(states), selected_action_log_probs, rewards
        

    def train_episode(self):
        states, selected_action_log_probs, rewards = self.generate_episode()

        T = len(rewards)

        discounted_returns = np.zeros(T)
        discounted_returns[T-1] = rewards[T-1]
        
        for t in range(T-2, -1, -1):
            discounted_returns[t] = rewards[t] + (self.gamma * discounted_returns[t+1])

        if self.args.use_baseline:
            state_values = self.state_value_network(torch.Tensor(states)).squeeze()
            deltas = torch.Tensor(discounted_returns) - state_values.detach()
        else:
            deltas = torch.Tensor(discounted_returns)

        policy_losses = []
        policy_loss_discount = 1

        for t in range(T):
            policy_loss = policy_loss_discount * deltas[t] * -selected_action_log_probs[t]
            policy_losses.append(policy_loss)
            policy_loss_discount *= self.gamma

        episode_policy_loss = torch.stack(policy_losses).sum()
        
        self.policy_optimizer.zero_grad()
        episode_policy_loss.backward()
        self.policy_optimizer.step()

        if self.args.use_baseline:
            state_value_loss = torch.dot(deltas, -state_values)
            self.state_value_optimizer.zero_grad()
            state_value_loss.backward()
            self.state_value_optimizer.step()

    
    def eval(self):
        ep_returns = []
        for ep in range(self.args.eval_episodes):
            ep_return = 0
            done = False
            observation, info = self.eval_env.reset()
            
            while not done:
                action_logits = self.policy_network(torch.Tensor(observation))
                action_probs = F.softmax(action_logits, dim=0).detach().numpy()
                action = np.random.choice(self.action_dim, p=action_probs)
                observation, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                ep_return += reward
               
            ep_returns.append(ep_return)

        return np.mean(ep_returns), np.std(ep_returns)


    def cleanup(self):
        self.env.close()
        self.eval_env.close()
        
            

class ReinforceNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=True),
        )
    
    def forward(self, x):
        return self.model(x)



if __name__ == "__main__":
    args = parse_args()
    exp_name = os.path.basename(__file__).rstrip(".py")
    run_name = f"{args.env_name}__{exp_name}__{args.env_seed}__{int(time.time())}"

    logging = args.use_tensorboard or args.use_wandb

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

    if logging:
        writer = SummaryWriter(log_dir=f"/Users/andrew/dev/deep-rl/runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    agent = ReinforceAgent(args, run_name)
    ep = 1
    t0 = time.time()
    while agent.global_timestep < args.total_timesteps:
        agent.train_episode()

        if ep % args.eval_freq == 0:
            t1 = time.time()
            print(f"\n{'=' * 16} EPISODE {ep} {'=' * 16}")
            print(f"Iteration time: {t1 - t0}")
            t0 = t1

            mean_return, std_return = agent.eval()
            if logging:
                writer.add_scalar("eval/mean_return", mean_return, global_step=ep)
                writer.add_scalar("eval/std_return", std_return, global_step=ep)

            print(f"Mean return: {mean_return}\nStd return: {std_return}\nGlobal timestep: {agent.global_timestep}")

        if args.save_model_freq > 0 and ep % args.save_model_freq == 0:
            model_path = f"/Users/andrew/dev/deep-rl/runs/{run_name}/{exp_name}__ep{str(ep)}.model"
            agent.save_model(model_path)

        ep += 1


    print(f"\n{'=' * 16} FINAL EVAL {'=' * 16}")
    mean_return, std_return = agent.eval()
    if logging:
        writer.add_scalar("eval/mean_return", mean_return,global_step=ep)
        writer.add_scalar("eval/std_return", std_return, global_step=ep)
        writer.close()

    print(f"Mean return: {mean_return}\nStd return: {std_return}\nGlobal timestep: {agent.global_timestep}")

    agent.cleanup()
        