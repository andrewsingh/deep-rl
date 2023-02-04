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
    parser.add_argument("--env-name", type=str, default="CartPole-v1")
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=500000)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--eval-num-episodes", type=int, default=10)
    parser.add_argument("--use-baseline", action='store_true')

    # Policy and baseline hyperparams
    parser.add_argument("--policy-hidden-dim", type=int, default=128)
    parser.add_argument("--baseline-hidden-dim", type=int, default=128)
    parser.add_argument("--policy-lr", type=float, default=2.5e-4)
    parser.add_argument("--baseline-lr", type=float, default=2.5e-4)
    parser.add_argument("--minibatch-size", type=int, default=128)

    # Other hyperparams
    parser.add_argument("--gamma", type=float, default=0.99)
    
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


# REINFORCE algorithm
def reinforce(args, env, eval_env, writer=None):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Device: {device}")

    observation, info = env.reset(seed=args.env_seed)
    state_dim = observation.shape[0]
    num_actions = env.action_space.n

    # Initialize policy and optional baseline
    policy_network = ReinforceNetwork(state_dim, num_actions, args.policy_hidden_dim).to(device)
    policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=args.policy_lr)

    if args.use_baseline:
        baseline_network = ReinforceNetwork(state_dim, 1, args.baseline_hidden_dim).to(device)
        baseline_optimizer = torch.optim.Adam(baseline_network.parameters(), lr=args.baseline_lr)

    global_timestep = 0
    ep = 0
    next_eval_timestep = args.eval_freq
    t0 = time.time()
    while global_timestep < args.total_timesteps:
        ep += 1
        done = False
        observation, info = env.reset()

        # Generate an episode
        states = []
        selected_action_log_probs = []
        rewards = []
        while not done:
            global_timestep += 1
            states.append(observation)
            action_logits = policy_network(torch.Tensor(observation).to(device))
            action_log_probs = F.log_softmax(action_logits, dim=0)
            action_probs = F.softmax(action_logits, dim=0).detach().cpu().numpy()
            action = np.random.choice(num_actions, p=action_probs)
            selected_action_log_probs.append(action_log_probs[action])

            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            rewards.append(reward)

        ep_len = len(rewards)

        # Calculate discounted returns
        discounted_returns = np.zeros(ep_len)
        discounted_returns[ep_len-1] = rewards[ep_len-1]
        for t in range(ep_len-2, -1, -1):
            discounted_returns[t] = rewards[t] + (args.gamma * discounted_returns[t+1])

        # Subtract state values from returns if using a baseline
        if args.use_baseline:
            state_values = baseline_network(torch.Tensor(np.array(states)).to(device)).squeeze()
            deltas = torch.Tensor(discounted_returns).to(device) - state_values.detach()
        else:
            deltas = torch.Tensor(discounted_returns).to(device)

        # Calculate policy loss for each timestep
        policy_losses = []
        policy_loss_discount = 1
        for t in range(ep_len):
            policy_loss = policy_loss_discount * deltas[t] * -selected_action_log_probs[t]
            policy_losses.append(policy_loss)
            policy_loss_discount *= args.gamma

        # Calculate overall policy loss and perform gradient descent step 
        episode_policy_loss = torch.stack(policy_losses).sum()
        policy_optimizer.zero_grad()
        episode_policy_loss.backward()
        policy_optimizer.step()

        # Calculate optional baseline loss and perform gradient descent step 
        if args.use_baseline:
            baseline_loss = torch.dot(deltas, -state_values)
            baseline_optimizer.zero_grad()
            baseline_loss.backward()
            baseline_optimizer.step()

        # Evaluation
        if global_timestep >= next_eval_timestep:
            t1 = time.time()
            print(f"\n{'=' * 16} TIMESTEP {global_timestep} {'=' * 16}")
            print(f"Iteration time: {t1 - t0}\nCurrent episode: {ep}")
            t0 = t1

            mean_return, std_return = evaluate(eval_env, policy_network, args.eval_num_episodes, device=device)

            if writer != None:
                writer.add_scalar("eval/mean_return", mean_return, global_step=global_timestep)
                writer.add_scalar("eval/std_return", std_return, global_step=global_timestep)

            print(f"Mean return: {mean_return}\nStd return: {std_return}")

            next_eval_timestep += args.eval_freq
            

def evaluate(eval_env, policy_network, num_episodes, device="cpu"):
    ep_returns = []
    for ep in range(num_episodes):
        ep_return = 0
        done = False
        observation, info = eval_env.reset()
        
        while not done:
            with torch.no_grad():
                action_logits = policy_network(torch.Tensor(observation).to(device))
                action_probs = F.softmax(action_logits, dim=0).cpu().numpy()
                action = np.random.choice(action_logits.shape[0], p=action_probs)

            observation, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            ep_return += reward
        
        ep_returns.append(ep_return)

    return np.mean(ep_returns), np.std(ep_returns)

      

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
    env = gym.make(args.env_name)
    eval_env = gym.make(args.env_name, render_mode='rgb_array')

    if args.record_video:
        video_folder = f"{root_dir}/runs/{run_name}/videos"
        os.makedirs(video_folder, exist_ok=True)
        eval_env = gym.wrappers.RecordVideo(eval_env, video_folder=video_folder, episode_trigger=lambda x: x % (args.eval_num_episodes * args.record_video_eval_freq) == 0)

    t_start = time.time()

    # Run REINFORCE algorithm
    reinforce(args, env, eval_env, writer=writer)

    t_end = time.time()
    print(f"Total time elapsed: {t_end - t_start}")

    # Cleanup
    env.close()
    eval_env.close()
    writer.close()
        