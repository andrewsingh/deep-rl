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
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-freq", type=int, default=10)
    parser.add_argument("--env-name", type=str, default="HalfCheetah-v4")
    parser.add_argument("--env-seed", type=int, default=42)
    
    parser.add_argument("--start-learning-timestep", type=int, default=10000)
    parser.add_argument("--actor-interval", type=int, default=2)
    parser.add_argument("--critic-interval", type=int, default=1)
    parser.add_argument("--update-target-interval", type=int, default=2)

    parser.add_argument("--actor-lr", type=float, default=2.5e-4)
    parser.add_argument("--critic-lr", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--action-noise-scale", type=float, default=0.2)
    parser.add_argument("--action-noise-clip", type=float, default=0.5)
    parser.add_argument("--actor-hidden-dim", type=int, default=64)
    parser.add_argument("--critic-hidden-dim", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=1000000)
    parser.add_argument("--minibatch-size", type=int, default=128)

    parser.add_argument("--use-tensorboard", action='store_true')
    parser.add_argument("--use-wandb", action='store_true')
    parser.add_argument("--wandb-project-name", type=str, default="deep-rl")
    parser.add_argument("--wandb-entity-name", type=str, default="andrew99")
    parser.add_argument("--save-model-freq", type=int, default=-1)
    parser.add_argument("--record-video", action='store_true')
    args = parser.parse_args()
    return args



class DDPGAgent():
    def __init__(self, args, run_name):
        self.args = args

        self.env = gym.make(self.args.env_name)
        self.eval_env = gym.make(self.args.env_name, render_mode='rgb_array')
        if self.args.record_video:
            video_folder = f"/Users/andrew/dev/deep-rl/runs/{run_name}/videos"
            os.makedirs(video_folder, exist_ok=True)
            self.eval_env = gym.wrappers.RecordVideo(self.eval_env, video_folder=video_folder, episode_trigger=lambda x: x % self.args.eval_episodes == 0)

        observation, info = self.env.reset(seed=args.env_seed)

        self.state_dim = observation.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        self.actor = ActorNetwork(self.state_dim, self.action_dim, self.args.actor_hidden_dim)
        self.critic = CriticNetwork(self.state_dim + self.action_dim, 1, self.args.critic_hidden_dim)
        
        self.actor_target = ActorNetwork(self.state_dim, self.action_dim, self.args.actor_hidden_dim)
        self.critic_target = CriticNetwork(self.state_dim + self.action_dim, 1, self.args.critic_hidden_dim)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, self.args.replay_capacity, self.args.minibatch_size)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.critic_lr)

        self.global_timestep = 0
 

    def train_episode(self):
        done = False
        observation, info = self.env.reset()
        while not done:
            self.global_timestep += 1

            if self.global_timestep < args.start_learning_timestep:
                action = self.env.action_space.sample()
            else:
                action = self.actor(torch.Tensor(observation)).detach().numpy()
                action_noise = np.clip(np.random.normal(loc=0.0, scale=self.args.action_noise_scale, size=self.action_dim), -self.args.action_noise_clip, self.args.action_noise_clip)
                action = np.clip(action + action_noise, -1, 1)
            
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            transition = (observation, action, reward, next_observation, int(terminated or truncated))
            self.replay_buffer.store_transition(transition)

            if self.global_timestep >= self.args.start_learning_timestep:
                if self.global_timestep % self.args.critic_interval == 0:
                    minibatch_states, minibatch_actions, minibatch_rewards,minibatch_next_states, minibatch_is_terminal = self.replay_buffer.sample_minibatch()
                    minibatch_is_terminal = minibatch_is_terminal.astype(bool)

                    with torch.no_grad():
                        minibatch_next_states_t = torch.Tensor(minibatch_next_states)
                        minibach_next_actions_t = self.actor_target(minibatch_next_states_t)

                        minibatch_target_q_values = self.critic_target(torch.cat((minibatch_next_states_t, minibach_next_actions_t), dim=1)).squeeze().numpy()
                        minibatch_q_targets = minibatch_rewards + (~minibatch_is_terminal * self.args.gamma * minibatch_target_q_values)
                
                    minibatch_q_preds = self.critic(torch.Tensor(np.concatenate((minibatch_states, minibatch_actions), axis=1))).squeeze()
                    
                    critic_loss = F.mse_loss(minibatch_q_preds, torch.Tensor(minibatch_q_targets), reduction='mean')

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()

                if self.global_timestep % self.args.actor_interval == 0:
                    minibatch_states_t = torch.Tensor(minibatch_states)
                    actor_action_preds = self.actor(minibatch_states_t)
                    critic_value_preds = self.critic(torch.cat((minibatch_states_t, actor_action_preds), dim=1))

                    actor_loss = -critic_value_preds.mean()
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                if self.global_timestep % self.args.update_target_interval == 0:
                    self.soft_update_target_network(self.actor_target, self.actor)
                    self.soft_update_target_network(self.critic_target, self.critic)

            observation = next_observation


    def soft_update_target_network(self, target_network, local_network):
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(self.args.tau * local_param.data + (1.0 - self.args.tau) * target_param.data)

    
    def eval(self):
        ep_returns = []
        for ep in range(self.args.eval_episodes):
            ep_return = 0
            done = False
            observation, info = self.eval_env.reset()
            
            while not done:
                with torch.no_grad():
                    action = self.actor(torch.Tensor(observation)).numpy()
                    
                observation, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                ep_return += reward
               
            ep_returns.append(ep_return)

        return np.mean(ep_returns), np.std(ep_returns)

        
    def save_model(self, model_path):
        torch.save(self.q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")


    def cleanup(self):
        self.env.close()
        self.eval_env.close()




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
        # curr_idx = 0
        # for elem in transition:
        #     transition_array[curr_idx : curr_idx + len(elem)] = elem
        #     curr_idx += len(elem)
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

    agent = DDPGAgent(args, run_name)
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
                mean_return, std_return = agent.eval()
                if logging:
                    writer.add_scalar("eval/mean_return", mean_return, global_step=agent.global_timestep)
                    writer.add_scalar("eval/std_return", std_return, global_step=agent.global_timestep)

                print(f"Mean return: {mean_return}\nStd return: {std_return}\nReplay buffer size: {agent.replay_buffer.current_size}\nGlobal timestep: {agent.global_timestep}")
            else:
                print(f"Skipping eval since haven't started training yet.\nReplay buffer size: {agent.replay_buffer.current_size}\nGlobal timestep: {agent.global_timestep}")

        if args.save_model_freq > 0 and ep % args.save_model_freq == 0:
            model_path = f"/Users/andrew/dev/deep-rl/runs/{run_name}/{exp_name}__ep{str(ep)}.model"
            agent.save_model(model_path)

        ep += 1


    print(f"\n{'=' * 16} FINAL EVAL {'=' * 16}")
    mean_return, std_return = agent.eval()
    if logging:
        writer.add_scalar("eval/mean_return", mean_return,global_step=agent.global_timestep)
        writer.add_scalar("eval/std_return", std_return, global_step=agent.global_timestep)
        writer.close()

    print(f"Mean return: {mean_return}\nStd return: {std_return}\nGlobal timestep: {agent.global_timestep}")

    agent.cleanup()