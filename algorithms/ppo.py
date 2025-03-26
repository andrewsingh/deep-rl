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
from torch.distributions.categorical import Categorical


def parse_args():
    parser = argparse.ArgumentParser()

    # Experiment settings
    parser.add_argument("--env-name", type=str, default="HalfCheetah-v5")
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument("--num-iters", type=int, default=None)
    parser.add_argument("--num-actors", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=250)
    parser.add_argument("--eval-freq", type=int, default=20000)
    parser.add_argument("--eval-num-episodes", type=int, default=10)

    # Policy and baseline hyperparams
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--value-coeff", type=float, default=0.5)
    parser.add_argument("--entropy-coeff", type=float, default=0.0)

    # Other hyperparams
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--normalize-advantages", action='store_true')
    parser.add_argument("--advantage-value-targets", action='store_true')
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    
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
    num_actions = envs.single_action_space.n
    is_discrete = envs.single_action_space.__class__.__name__ == "Discrete"

    print(f"Env name: {args.env_name}\nAction space: {envs.single_action_space}\nAction space type: {'Discrete' if is_discrete else 'Continuous'}")

    if is_discrete:
        agent = DiscreteAgent(state_dim[0], num_actions, args.hidden_dim).to(device)
    else:
        agent = ContinuousAgent(state_dim[0], action_dim[0], args.hidden_dim)

    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)

    global_timestep = 0
    next_eval_timestep = args.eval_freq

    total_steps_per_iter = args.num_actors * args.rollout_steps

    num_iters = args.num_iters
    if num_iters is None:
        num_iters = int(np.ceil(args.total_timesteps / total_steps_per_iter))
        print(f"Calculated {num_iters} iters based on {args.total_timesteps} total timesteps")

    obs, infos = envs.reset(seed=args.env_seed)
    for iter in range(num_iters):
        t0 = time.time()
        rollout_states = torch.zeros((args.rollout_steps, args.num_actors) + state_dim).to(device)
        rollout_actions = torch.zeros((args.rollout_steps, args.num_actors) + (action_dim if not is_discrete else ())).to(device)
        rollout_logprobs = torch.zeros((args.rollout_steps, args.num_actors)).to(device)
        rollout_entropies = torch.zeros((args.rollout_steps, args.num_actors)).to(device)
        rollout_values = torch.zeros((args.rollout_steps, args.num_actors)).to(device)
        rollout_rewards = torch.zeros((args.rollout_steps, args.num_actors)).to(device)
        rollout_dones = torch.zeros((args.rollout_steps, args.num_actors)).to(device)
        
        # print("Collecting rollouts")
        r0 = time.time()
        for step in range(args.rollout_steps):
            global_timestep += args.num_actors
            obs = torch.from_numpy(obs).float().to(device)
            rollout_states[step] = obs
            actions, logprobs, entropies = agent.get_action(obs)
            values = agent.get_value(obs)
            obs, rewards, terms, truncs, infos = envs.step(actions.cpu().numpy())
            dones = terms | truncs
            rollout_actions[step] = actions
            rollout_logprobs[step] = logprobs
            rollout_entropies[step] = entropies
            rollout_values[step] = values.squeeze()
            rollout_rewards[step] = torch.from_numpy(rewards).float().to(device)
            rollout_dones[step] = torch.from_numpy(dones).float().to(device)
        
        last_obs = torch.from_numpy(obs).float().to(device)
        last_values = agent.get_value(last_obs).squeeze()
        rollout_advantages =  torch.zeros((args.rollout_steps, args.num_actors))
        rollout_value_targets =  torch.zeros((args.rollout_steps, args.num_actors))

        next_non_terminal = 1.0 - rollout_dones[-1]
        rollout_advantages[-1] = (rollout_rewards[-1] + (args.gamma * last_values * next_non_terminal) - rollout_values[-1])
        rollout_value_targets[-1] = rollout_rewards[-1] + (args.gamma * last_values * next_non_terminal)
        
        for step in range(args.rollout_steps - 2, -1, -1):
            next_non_terminal = 1.0 - rollout_dones[step]
            delta = rollout_rewards[step] + (args.gamma * rollout_values[step + 1] * next_non_terminal) - rollout_values[step]
            rollout_advantages[step] = delta + (args.gamma * args.gae_lambda * next_non_terminal) * rollout_advantages[step + 1]
            rollout_value_targets[step] = rollout_rewards[step] + (args.gamma * rollout_value_targets[step + 1] * next_non_terminal)

        if args.advantage_value_targets:
            rollout_value_targets = rollout_advantages + rollout_values
        
        data_states = rollout_states.reshape((-1,) + state_dim)
        data_actions = rollout_actions.reshape(((-1,) + action_dim) if not is_discrete else -1)
        data_logprobs = rollout_logprobs.reshape(-1).detach()
        data_advantages = rollout_advantages.reshape(-1).detach()
        if args.normalize_advantages:
            data_advantages = (data_advantages - data_advantages.mean()) / (data_advantages.std() + 1e-8)
        data_value_targets = rollout_value_targets.reshape(-1).detach()
        data_size = args.rollout_steps * args.num_actors
        
        num_batches = data_size // args.batch_size
        clip_ratios_by_epoch = {}
        divergence_by_epoch = {}

        r1 = time.time()

        if writer != None:
            writer.add_scalar("train/mean_reward", rollout_rewards.mean(), global_step=global_timestep)
            writer.add_scalar("train/std_reward", rollout_rewards.std(), global_step=global_timestep)
            writer.add_scalar("train/mean_value_target", rollout_value_targets.mean(), global_step=global_timestep)
            writer.add_scalar("train/std_value_target", rollout_value_targets.std(), global_step=global_timestep)
    
        # Optimization
        for epoch in range(args.epochs):
            clip_ratios_by_epoch[epoch] = []
            divergence_by_epoch[epoch] = []
            indices = torch.randperm(data_size)
            for i in range(num_batches):
                batch_indices = indices[i * args.batch_size : (i + 1) * args.batch_size]
                batch_states = data_states[batch_indices]
                batch_actions = data_actions[batch_indices]
                batch_advantages = data_advantages[batch_indices]
                batch_old_logprobs = data_logprobs[batch_indices]
                _, batch_logprobs, batch_entropies = agent.get_action(batch_states, action=batch_actions)
                batch_ratios = torch.exp(batch_logprobs - batch_old_logprobs)
                divergence_by_epoch[epoch].append(torch.abs(1.0 - batch_ratios).mean().detach())
                batch_clipped_ratios = torch.clamp(batch_ratios, min=1-args.clip_eps, max=1+args.clip_eps)
                clip_ratio = (batch_ratios != batch_clipped_ratios).sum() / len(batch_ratios)
                clip_ratios_by_epoch[epoch].append(clip_ratio)

                actor_loss = -torch.min(batch_ratios * batch_advantages, batch_clipped_ratios * batch_advantages).mean()
    
                entropy = batch_entropies.mean()

                batch_values = agent.get_value(batch_states).squeeze()
                batch_value_targets = data_value_targets[batch_indices]
                critic_loss = F.mse_loss(batch_values, batch_value_targets, reduction='mean')

                overall_loss = actor_loss + (args.value_coeff * critic_loss) - (args.entropy_coeff * entropy)
                
                optimizer.zero_grad()
                overall_loss.backward()
                grad_norm_before_clip = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                if writer != None:
                    writer.add_scalar("train/actor_loss", actor_loss, global_step=global_timestep)
                    writer.add_scalar("train/critic_loss", critic_loss, global_step=global_timestep)
                    writer.add_scalar("train/entropy", entropy, global_step=global_timestep)
                    writer.add_scalar("train/grad_norm_before_clip", grad_norm_before_clip, global_step=global_timestep)
                    writer.add_scalar("train/batch_values", batch_values.mean(), global_step=global_timestep)
                    writer.add_scalar("train/batch_value_targets", batch_value_targets.mean(), global_step=global_timestep)
                    

        r2 = time.time()
        

        # Evaluation
        if global_timestep >= next_eval_timestep:
            t1 = time.time()
            print(f"\n{'=' * 16} TIMESTEP {global_timestep} - ITERATION {iter + 1} {'=' * 16}")
            print(f"Iteration time: {t1 - t0}")
            
            mean_return, std_return = evaluate(eval_env, agent, args.eval_num_episodes)

            if writer != None:
                writer.add_scalar("eval/mean_return", mean_return, global_step=global_timestep)
                writer.add_scalar("eval/std_return", std_return, global_step=global_timestep)

            print(f"Mean return: {mean_return}\nStd return: {std_return}")
                
            next_eval_timestep += args.eval_freq

            print(f"Rollout time: {r1 - r0}")
            print(f"Optimization time: {r2 - r1}")
            print(f"Ratio: {(r1 - r0) / (r2 - r1)}")
            
            if writer != None:
                writer.add_scalar("train/mean_clip_fraction_first_epoch", np.mean(clip_ratios_by_epoch[0]), global_step=global_timestep)
                writer.add_scalar("train/mean_clip_fraction_last_epoch", np.mean(clip_ratios_by_epoch[args.epochs - 1]), global_step=global_timestep)
                writer.add_scalar("train/mean_divergence_first_epoch", np.mean(divergence_by_epoch[0]), global_step=global_timestep)
                writer.add_scalar("train/mean_divergence_last_epoch", np.mean(divergence_by_epoch[args.epochs - 1]), global_step=global_timestep)

            
            

def evaluate(eval_env, agent, num_episodes, device="cpu"):
    ep_returns = []
    for ep in range(num_episodes):
        ep_return = 0
        done = False
        obs, info = eval_env.reset()

        while not done:
            with torch.no_grad():
                action, _, _ = agent.get_action(torch.Tensor(obs).unsqueeze(dim=0).to(device))

            obs, reward, terminated, truncated, info = eval_env.step(action.squeeze(dim=0).cpu().numpy())
            done = terminated or truncated
            ep_return += reward
        
        ep_returns.append(ep_return)

    return np.mean(ep_returns), np.std(ep_returns)

      

class DiscreteAgent(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim, bias=True),
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=True),
        )

    def get_action(self, x, action=None):
        action_logits = self.actor(x)
        probs = Categorical(logits=action_logits)
        
        if action is None:
            action = probs.sample()
       
        return action, probs.log_prob(action), probs.entropy()

    def get_value(self, x):
        return self.critic(x)
    

class ContinuousAgent(nn.Module):
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
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=True),
        )

    def get_action(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
       
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)

    def get_value(self, x):
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
        