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
from torch.distributions.normal import Normal
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    # Experiment settings
    parser.add_argument("--env-name", type=str, default="Pendulum-v1")
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--num-iters", type=int, default=1)
    parser.add_argument("--num-actors", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=100)
    parser.add_argument("--eval-freq", type=int, default=5)
    parser.add_argument("--eval-num-episodes", type=int, default=10)

    # Policy and baseline hyperparams
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--actor-lr", type=float, default=2.5e-4)
    parser.add_argument("--critic-lr", type=float, default=2.5e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)

    # Other hyperparams
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    
    # Logging
    parser.add_argument("--cpu", action='store_true')
    parser.add_argument("--use-tensorboard", action='store_true')
    parser.add_argument("--use-wandb", action='store_true')
    parser.add_argument("--wandb-project-name", type=str, default="deep-rl")
    parser.add_argument("--wandb-entity-name", type=str, default="andrew99")
    parser.add_argument("--record-video", action='store_true')
    parser.add_argument("--record-video-eval-freq", type=int, default=2)

    args = parser.parse_args()
    return args


# PPO algorithm
def ppo(args, envs, eval_env, writer=None):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Device: {device}")

  
    state_dim = envs.single_observation_space.shape
    action_dim = envs.single_action_space.shape

    actor = Actor(state_dim[0], action_dim[0], args.hidden_dim).to(device)
    critic = Critic(state_dim[0], 1, args.hidden_dim).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    global_timestep = 0
    
    for iter in range(args.num_iters):
        t0 = time.time()
        obs, infos = envs.reset(seed=args.env_seed)
        rollout_states = torch.zeros((args.rollout_steps, args.num_actors) + state_dim).to(device)
        rollout_actions =  torch.zeros((args.rollout_steps, args.num_actors) + action_dim).to(device)
        rollout_logprobs =  torch.zeros((args.rollout_steps, args.num_actors)).to(device)
        rollout_values =  torch.zeros((args.rollout_steps, args.num_actors)).to(device)
        rollout_rewards =  torch.zeros((args.rollout_steps, args.num_actors)).to(device)
        rollout_dones =  torch.zeros((args.rollout_steps, args.num_actors)).to(device)
        
        print("Collecting rollouts")
        for step in range(args.rollout_steps):
            global_timestep += args.num_actors
            obs = torch.from_numpy(obs).float().to(device)
            rollout_states[step] = obs
            actions, logprobs = actor(obs)
            values = critic(obs)
            # pdb.set_trace()
            obs, rewards, terms, truncs, infos = envs.step(actions.cpu().numpy())
            dones = terms | truncs
            rollout_actions[step] = actions
            rollout_logprobs[step] = logprobs
            rollout_values[step] = values.squeeze()
            rollout_rewards[step] = torch.from_numpy(rewards).float().to(device)
            rollout_dones[step] = torch.from_numpy(dones).float().to(device)
        
        last_obs = torch.from_numpy(obs).float().to(device)
        last_values = critic(last_obs).squeeze()
        rollout_advantages =  torch.zeros((args.rollout_steps, args.num_actors))
        rollout_value_targets =  torch.zeros((args.rollout_steps, args.num_actors))

        next_non_terminal = 1.0 - rollout_dones[-1]
        rollout_advantages[-1] = (rewards[-1] + (args.gamma * last_values *next_non_terminal) - values[-1])
        rollout_value_targets[-1] = rewards[-1] + (args.gamma * last_values * next_non_terminal)
        
        for step in range(args.rollout_steps - 2, -1, -1):
            
            next_non_terminal = 1.0 - rollout_dones[step]
            delta = rollout_rewards[step] + (args.gamma * rollout_values[step + 1] *next_non_terminal) - rollout_values[step]
            rollout_advantages[step] = delta + (args.gamma * args.gae_lambda *next_non_terminal) * rollout_advantages[step + 1]
            rollout_value_targets[step] = rollout_rewards[step] + (args.gamma * rollout_value_targets[step + 1] * next_non_terminal)

        data_states = rollout_states.reshape((-1,) + state_dim)
        data_actions = rollout_actions.reshape((-1,) + action_dim)
        data_logprobs = rollout_logprobs.reshape(-1).detach()
        data_advantages = rollout_advantages.reshape(-1).detach()
        data_value_targets = rollout_value_targets.reshape(-1).detach()
        data_size = args.rollout_steps * args.num_actors
        
        num_batches = data_size // args.batch_size

    
        # Optimization
        print("Optimizing actor and critic")
        for epoch in tqdm(range(args.epochs)):
            indices = torch.randperm(data_size)
            for i in range(num_batches):
                batch_indices = indices[i * args.batch_size : (i + 1) * args.batch_size]
                batch_states = data_states[batch_indices]
                batch_actions = data_actions[batch_indices]
                batch_advantages = data_advantages[batch_indices]
                batch_old_logprobs = data_logprobs[batch_indices]
                _, batch_logprobs = actor(batch_states, action=batch_actions)
                batch_ratios = torch.exp(batch_logprobs - batch_old_logprobs)
                batch_clipped_ratios = torch.clamp(batch_ratios, min=1-args.clip_eps, max=1+args.clip_eps)
                actor_loss = -torch.min(batch_ratios * batch_advantages, batch_clipped_ratios * batch_advantages).mean()

                batch_values = critic(batch_states).squeeze()
                batch_value_targets = data_value_targets[batch_indices]
                critic_loss = F.mse_loss(batch_values, batch_value_targets, reduction='mean')

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                
        # Evaluation
        # if global_timestep >= next_eval_timestep:
        t1 = time.time()
        print(f"\n{'=' * 16} ITERATION {iter + 1} {'=' * 16}")
        print(f"Iteration time: {t1 - t0}")

        if iter == 0 or (iter + 1) % args.eval_freq == 0:
            mean_return, std_return = evaluate(eval_env, actor, args.eval_num_episodes)

            if writer != None:
                writer.add_scalar("eval/mean_return", mean_return, global_step=global_timestep)
                writer.add_scalar("eval/std_return", std_return, global_step=global_timestep)

            print(f"Mean return: {mean_return}\nStd return: {std_return}")

        # next_eval_timestep += args.eval_freq
            

def evaluate(eval_env, actor, num_episodes, device="cpu"):
    ep_returns = []
    for ep in range(num_episodes):
        ep_return = 0
        done = False
        obs, info = eval_env.reset()

        while not done:
            with torch.no_grad():
                action, _ = actor(torch.Tensor(obs).unsqueeze(dim=0).to(device))

            obs, reward, terminated, truncated, info = eval_env.step(action.squeeze(dim=0).cpu().numpy())
            done = terminated or truncated
            ep_return += reward
        
        ep_returns.append(ep_return)

    return np.mean(ep_returns), np.std(ep_returns)

      
    

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim, bias=True),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, output_dim))
    
    def forward(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
       
        return action, probs.log_prob(action).sum(1)


class Critic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim, bias=True),
        )
    
    def forward(self, x):
        return self.critic(x)


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
    envs = gym.make_vec(args.env_name, num_envs=args.num_actors, vectorization_mode="sync")
    eval_env = gym.make(args.env_name, render_mode='rgb_array')

    if args.record_video:
        video_folder = f"{root_dir}/runs/{run_name}/videos"
        os.makedirs(video_folder, exist_ok=True)
        eval_env = gym.wrappers.RecordVideo(eval_env, video_folder=video_folder, episode_trigger=lambda x: x % (args.eval_num_episodes * args.record_video_eval_freq) == 0)

    t_start = time.time()

    # Run PPO algorithm
    ppo(args, envs, eval_env, writer=writer)

    t_end = time.time()
    print(f"Total time elapsed: {t_end - t_start}")

    # Cleanup
    envs.close()
    eval_env.close()
    if writer:
        writer.close()
        