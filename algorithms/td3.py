import gymnasium as gym
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
import os
import time
import pdb

from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser()

    # Experiment settings
    parser.add_argument("--env-name", type=str, default="HalfCheetah-v2")
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--eval-num-episodes", type=int, default=10)

    # Network hyperparams
    parser.add_argument("--actor-hidden-dim", type=int, default=128)
    parser.add_argument("--critic-hidden-dim", type=int, default=128)
    parser.add_argument("--actor-lr", type=float, default=2.5e-4)
    parser.add_argument("--critic-lr", type=float, default=2.5e-4)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--actor-freq", type=int, default=2)
    parser.add_argument("--update-target-freq", type=int, default=2)

    # Other hyperparams
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--exploration-noise-scale", type=float, default=0.1)
    parser.add_argument("--exploration-noise-clip", type=float, default=0.5)
    parser.add_argument("--target-noise-scale", type=float, default=0.1)
    parser.add_argument("--target-noise-clip", type=float, default=0.5)
    parser.add_argument("--replay-capacity", type=int, default=1000000)
    parser.add_argument("--start-learning-timestep", type=int, default=25000)
    
    # Logging
    parser.add_argument("--cpu", action='store_true')
    parser.add_argument("--use-tensorboard", action='store_true')
    parser.add_argument("--use-wandb", action='store_true')
    parser.add_argument("--wandb-project-name", type=str, default="deep-rl")
    parser.add_argument("--wandb-entity-name", type=str, default="andrew99")
    parser.add_argument("--record-video", action='store_true')

    args = parser.parse_args()
    return args



def td3(args, env, eval_env, writer=None):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Device: {device}")

    observation, info = env.reset(seed=args.env_seed)
    state_dim = observation.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize actor and critic networks and replay buffer
    actor = ActorNetwork(state_dim, action_dim, args.actor_hidden_dim).to(device)
    critic1 = CriticNetwork(state_dim + action_dim, 1, args.critic_hidden_dim).to(device)
    critic2 = CriticNetwork(state_dim + action_dim, 1, args.critic_hidden_dim).to(device)
    
    actor_target = ActorNetwork(state_dim, action_dim, args.actor_hidden_dim).to(device)
    critic1_target = CriticNetwork(state_dim + action_dim, 1, args.critic_hidden_dim).to(device)
    critic2_target = CriticNetwork(state_dim + action_dim, 1, args.critic_hidden_dim).to(device)

    actor_target.load_state_dict(actor.state_dict())
    critic1_target.load_state_dict(critic1.state_dict())
    critic2_target.load_state_dict(critic2.state_dict())

    replay_buffer = ReplayBuffer(state_dim, action_dim, args.replay_capacity, args.minibatch_size)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optimizer = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optimizer = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

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

            # Select action
            if global_timestep < args.start_learning_timestep:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = actor(torch.Tensor(observation).to(device)).cpu().numpy()
                    action_noise = np.random.normal(loc=0.0, scale=args.exploration_noise_scale, size=action_dim).clip(-args.exploration_noise_scale, args.exploration_noise_scale)
                    action = (action + action_noise).clip(-1, 1)
            
            # Execute action in environment and store transition in replay buffer
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            transition = (observation, action, reward, next_observation, int(terminated or truncated))
            replay_buffer.store_transition(transition)

            if global_timestep >= args.start_learning_timestep:
                # Sample random minibatch of transitions from replay buffer
                minibatch_states, minibatch_actions, minibatch_rewards,minibatch_next_states,minibatch_is_terminal = replay_buffer.sample_minibatch()
                minibatch_is_terminal = minibatch_is_terminal.astype(bool)

                # Calculate critic targets
                with torch.no_grad():
                    minibatch_next_states_t = torch.Tensor(minibatch_next_states).to(device)
                    minibatch_next_actions = actor_target(minibatch_next_states_t)
                    minibatch_next_actions_noise = torch.normal(mean=0.0, std=args.target_noise_scale, size=minibatch_next_actions.shape).clamp(-args.target_noise_clip, args.target_noise_clip).to(device)
                    minibatch_next_actions = (minibatch_next_actions + minibatch_next_actions_noise).clamp(-1, 1)

                    minibatch_target_q_values1 = critic1_target(torch.cat((minibatch_next_states_t, minibatch_next_actions), dim=1)).squeeze().cpu().numpy()
                    minibatch_target_q_values2 = critic2_target(torch.cat((minibatch_next_states_t, minibatch_next_actions), dim=1)).squeeze().cpu().numpy()

                    # Take the minimum of the two critics (clipped double Q-learning)
                    minibatch_target_q_values = np.minimum(minibatch_target_q_values1, minibatch_target_q_values2)
                    minibatch_q_targets = minibatch_rewards + (~minibatch_is_terminal * args.gamma * minibatch_target_q_values)
                    minibatch_q_targets_t = torch.Tensor(minibatch_q_targets).to(device)
            
                # Update each of the critics
                for critic, critic_optimizer in [(critic1, critic1_optimizer), (critic2, critic2_optimizer)]:
                    # Get critic predictions 
                    minibatch_q_preds = critic(torch.Tensor(np.concatenate((minibatch_states, minibatch_actions), axis=1)).to(device)).squeeze()

                    # Calculate critic loss and perform gradient descent step
                    critic_loss = F.mse_loss(minibatch_q_preds, minibatch_q_targets_t, reduction='mean')
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    critic_optimizer.step()

                # Delayed policy update
                if global_timestep % args.actor_freq == 0:
                    # Get first critic's value predictions of actor's action predictions
                    minibatch_states_t = torch.Tensor(minibatch_states).to(device)
                    actor_action_preds = actor(minibatch_states_t)
                    critic1_value_preds = critic1(torch.cat((minibatch_states_t, actor_action_preds), dim=1).to(device))

                    # Calculate actor loss and perform gradient descent step
                    actor_loss = -critic1_value_preds.mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                # Delayed target network updates
                if global_timestep % args.update_target_freq == 0:
                    soft_update_target_network(actor_target, actor, args.tau)
                    soft_update_target_network(critic1_target, critic, args.tau)
                    soft_update_target_network(critic2_target, critic, args.tau)

            observation = next_observation

            # Evaluation
            if global_timestep % args.eval_freq == 0:
                t1 = time.time()
                print(f"\n{'=' * 16} TIMESTEP {global_timestep} {'=' * 16}")
                print(f"Iteration time: {t1 - t0}\nCurrent episode: {ep}")
                t0 = t1

                if global_timestep >= args.start_learning_timestep:
                    mean_return, std_return = evaluate(eval_env, actor, args.eval_num_episodes, device=device)

                    if writer != None:
                        writer.add_scalar("eval/mean_return", mean_return, global_step=global_timestep)
                        writer.add_scalar("eval/std_return", std_return, global_step=global_timestep)

                    print(f"Mean return: {mean_return}\nStd return: {std_return}\nReplay buffer size: {replay_buffer.current_size}")
                else:
                    print(f"Skipping eval - still filling the replay buffer.\nReplay buffer size: {replay_buffer.current_size}")


def evaluate(eval_env, actor, num_episodes, device="cpu"):
    ep_returns = []
    for ep in range(num_episodes):
        ep_return = 0
        done = False
        observation, info = eval_env.reset()
        
        while not done:
            with torch.no_grad():
                action = actor(torch.Tensor(observation).to(device)).cpu().numpy()
                
            observation, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            ep_return += reward
        
        ep_returns.append(ep_return)

    return np.mean(ep_returns), np.std(ep_returns)
            

def soft_update_target_network(target_network, local_network, tau):
    for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)



class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=True),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.model(x)



class CriticNetwork(nn.Module):
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



class ReplayBuffer():
    def __init__(self, state_dim, action_dim, capacity, minibatch_size):
        self.transition_dim = (2 * state_dim) + action_dim + 2
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.current_size = 0
        self.minibatch_size = minibatch_size
        self.next_store_idx = 0
        self.buffer = np.zeros((self.capacity, self.transition_dim))
    
    def store_transition(self, transition):
        (observation, action, reward, next_observation, done) = transition
        transition_array = np.zeros(self.transition_dim)
        transition_array[:self.state_dim] = observation
        transition_array[self.state_dim : self.state_dim + self.action_dim] = action
        transition_array[self.state_dim + self.action_dim] = reward
        transition_array[self.state_dim + self.action_dim + 1 : (2 * self.state_dim) + self.action_dim + 1] = next_observation
        transition_array[(2 * self.state_dim) + self.action_dim + 1] = done
        self.buffer[self.next_store_idx] = transition_array
        self.next_store_idx = (self.next_store_idx + 1) % self.capacity
        self.current_size = min(self.current_size + 1, self.capacity)

    def sample_minibatch(self):
        indices = np.random.choice(self.current_size, self.minibatch_size)
        minibatch = self.buffer[indices]
        minibatch_states = minibatch[:, :self.state_dim]
        minibatch_actions = minibatch[:, self.state_dim : self.state_dim + self.action_dim]
        minibatch_rewards = minibatch[:, self.state_dim + self.action_dim]
        minibatch_next_states = minibatch[:, self.state_dim + self.action_dim + 1 : (2 * self.state_dim) + self.action_dim + 1]
        minibatch_is_terminal = minibatch[:, (2 * self.state_dim) + self.action_dim + 1]
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
    env = gym.make(args.env_name)
    eval_env = gym.make(args.env_name, render_mode='rgb_array')

    if args.record_video:
        video_folder = f"{root_dir}/runs/{run_name}/videos"
        os.makedirs(video_folder, exist_ok=True)
        eval_env = gym.wrappers.RecordVideo(eval_env, video_folder=video_folder, episode_trigger=lambda x: x % args.eval_num_episodes == 0)

    t_start = time.time()

    # Run TD3 algorithm
    td3(args, env, eval_env, writer=writer)

    t_end = time.time()
    print(f"Total time elapsed: {t_end - t_start}")

    # Cleanup
    env.close()
    eval_env.close()
    writer.close()

    

