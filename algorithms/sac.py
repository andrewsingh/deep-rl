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
from torch.distributions.normal import Normal



def parse_args():
    parser = argparse.ArgumentParser()

    # Experiment settings
    parser.add_argument("--env-name", type=str, default="HalfCheetah-v5")
    parser.add_argument("--env-seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--eval-num-episodes", type=int, default=10)
    parser.add_argument("--reward-scale", type=int, default=5)

    # Network hyperparams
    parser.add_argument("--actor-hidden-dim", type=int, default=256)
    parser.add_argument("--critic-hidden-dim", type=int, default=256)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--value-lr", type=float, default=2.5e-4)
    parser.add_argument("--minibatch-size", type=int, default=256)

    # Other hyperparams
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--replay-capacity", type=int, default=1000000)
    parser.add_argument("--start-learning-timestep", type=int, default=25000)
    parser.add_argument("--grad-steps-per-env-step", type=int, default=1)
    parser.add_argument("--update-target-freq", type=int, default=1)
    
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


# SAC algorithm
def sac(args, env, eval_env, writer=None):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Device: {device}")

    actor_updates = 0

    observation, info = env.reset(seed=args.env_seed)
    state_dim = observation.shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = (env.action_space.high - env.action_space.low) / 2.0
    action_bias = (env.action_space.high + env.action_space.low) / 2.0

    # Initialize actor and critic networks and replay buffer
    actor = ActorNetwork(state_dim, action_dim, args.actor_hidden_dim, action_scale, action_bias).to(device)
    critic1 = CriticNetwork(state_dim + action_dim, 1, args.critic_hidden_dim).to(device)
    critic2 = CriticNetwork(state_dim + action_dim, 1, args.critic_hidden_dim).to(device)
    critic1_target = CriticNetwork(state_dim + action_dim, 1, args.critic_hidden_dim).to(device)
    critic2_target = CriticNetwork(state_dim + action_dim, 1, args.critic_hidden_dim).to(device)
    
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
            ts0 = time.time()
            global_timestep += 1

            # Select action
            if global_timestep < args.start_learning_timestep:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action, _ = actor(torch.Tensor(observation).unsqueeze(dim=0).to(device))
                    action = action.squeeze(dim=0).cpu().numpy()
                    
            # Execute action in environment and store transition in replay buffer
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            reward_scaled = reward * args.reward_scale
            transition = (observation, action, reward_scaled, next_observation, int(terminated or truncated))
            replay_buffer.store_transition(transition)

            ts1 = time.time()
            # print("ts 0 to 1: ", ts1 - ts0)

            if global_timestep >= args.start_learning_timestep:
                for _ in range(args.grad_steps_per_env_step):
                    ts2 = time.time()
                    # Sample random minibatch of transitions from replay buffer
                    minibatch_states, minibatch_actions, minibatch_rewards, minibatch_next_states, minibatch_is_terminal = replay_buffer.sample_minibatch()
                    minibatch_states_t = torch.Tensor(minibatch_states).to(device)
                    minibatch_next_states_t = torch.Tensor(minibatch_next_states).to(device)
                    minibatch_actions_t = torch.Tensor(minibatch_actions).to(device)

                    ts3 = time.time()
                    # print("ts 2 to 3: ", ts3 - ts2)

                    # Update Q networks
                    with torch.no_grad():
                        minibatch_action_preds, minibatch_action_pred_logprobs = actor(minibatch_states_t)
                        minibatch_next_action_preds, minibatch_next_action_pred_logprobs = actor(minibatch_next_states_t)
                        minibatch_state_action_t = torch.cat((minibatch_states_t, minibatch_actions_t), dim=1)
                        minibatch_next_state_action_pred_t = torch.cat((minibatch_next_states_t, minibatch_next_action_preds), dim=1)
                        minibatch_target_q_values = torch.min(critic1_target(minibatch_next_state_action_pred_t), critic2_target(minibatch_next_state_action_pred_t)).squeeze(dim=1).cpu().numpy()
                        minibatch_q_targets = minibatch_rewards + ((1 - minibatch_is_terminal) * args.gamma * (minibatch_target_q_values - (args.alpha * minibatch_next_action_pred_logprobs.cpu().numpy())))
                        minibatch_q_targets_t = torch.Tensor(minibatch_q_targets).to(device)

                    for idx, (critic, critic_optimizer) in enumerate([(critic1, critic1_optimizer), (critic2, critic2_optimizer)]):
                        minibatch_q_preds = critic(minibatch_state_action_t)
                        critic_loss = F.mse_loss(minibatch_q_preds.squeeze(dim=1), minibatch_q_targets_t, reduction='mean')
                        critic_optimizer.zero_grad()
                        critic_loss.backward()
                        critic_optimizer.step()

                        if writer != None:
                            writer.add_scalar(f"train/critic{str(idx + 1)}_loss", critic_loss, global_step=global_timestep)
                    
                    ts5 = time.time()
                    # print("ts 4 to 5: ", ts5 - ts4)

                    # Update policy network
                    minibatch_action_preds, minibatch_action_pred_logprobs = actor(minibatch_states_t)
                    minibatch_state_action_pred_t = torch.cat((minibatch_states_t, minibatch_action_preds), dim=1)
                    minibatch_q_values = torch.min(critic1(minibatch_state_action_pred_t), critic2(minibatch_state_action_pred_t)).squeeze(dim=1)
                    actor_loss = ((args.alpha * minibatch_action_pred_logprobs) - minibatch_q_values).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    actor_updates += 1

                    if writer != None:
                        writer.add_scalar("train/actor_loss", actor_loss, global_step=global_timestep)

                    # Update target value network
                    if global_timestep % args.update_target_freq == 0:
                        soft_update_target_network(critic1_target, critic1, args.tau)
                        soft_update_target_network(critic2_target, critic2, args.tau)
                    
                    ts6 = time.time()
                    # print("ts 5 to 6: ", ts6 - ts5)
                    # print("\n\n")
                    
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
                    print(f"Timesteps per actor update: {(global_timestep - args.start_learning_timestep) / actor_updates}")
                    print(f"Time elapsed: {time.time() - t_start}")
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
                action, _ = actor(torch.Tensor(observation).unsqueeze(dim=0).to(device))
                action = action.squeeze(dim=0).cpu().numpy()
                
            observation, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            ep_return += reward
        
        ep_returns.append(ep_return)

    return np.mean(ep_returns), np.std(ep_returns)
            

def soft_update_target_network(target_network, local_network, tau):
    for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)



class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, action_scale, action_bias):
        super().__init__()
        self.register_buffer("action_scale", torch.Tensor(action_scale))
        self.register_buffer("action_bias", torch.Tensor(action_bias))
        self.actor_mean = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=True),
            nn.Tanh()
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, output_dim))
    
    def forward(self, x, eval=False):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        epsilon = torch.randn_like(action_mean)
        action_raw = action_mean + ((not eval) * action_std * epsilon)
        action_squashed = torch.tanh(action_raw)

        probs = Normal(action_mean, action_std)
        action_raw_logprob = probs.log_prob(action_raw).sum(1)
        action_squashed_logprob = action_raw_logprob - torch.log(1 - action_squashed.pow(2) + 1e-6).sum(1)

        return (action_squashed * self.action_scale) + self.action_bias, action_squashed_logprob
    
   
       

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
        eval_env = gym.wrappers.RecordVideo(eval_env, video_folder=video_folder, episode_trigger=lambda x: x % (args.eval_num_episodes * args.record_video_eval_freq) == 0)

    t_start = time.time()

    # Run SAC algorithm
    sac(args, env, eval_env, writer=writer)

    t_end = time.time()
    print(f"Total time elapsed: {t_end - t_start}")

    # Cleanup
    env.close()
    eval_env.close()
    writer.close()

    

