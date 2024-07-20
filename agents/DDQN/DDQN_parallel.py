from game.snake_game import SnakeGameAI

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import pygame
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv

# parallel version of DDQN
from concurrent.futures import ThreadPoolExecutor, as_completed


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))
        # print(f"Added to replay buffer: {len(self.buffer)} / {self.buffer.maxlen}")  # Debugging

    def sample(self, batch_size):
        samples = random.sample(self.buffer, min(len(self.buffer), batch_size))
        state, action, reward, next_state, done = zip(*samples)
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

class DuelingQNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, embedding_dim=32, dropout=0.2):
        super(DuelingQNetwork, self).__init__()
        self.embedding = nn.Embedding(6, embedding_dim)
        self.fc1 = nn.Linear(input_dims * embedding_dim, 2048)
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(2048, 1024)
        self.dropout2 = nn.Dropout(p=dropout)
        self.fc3 = nn.Linear(1024, 512)
        self.dropout3 = nn.Dropout(p=dropout)
        self.fc_value = nn.Linear(512, 256)
        self.value = nn.Linear(256, 1)
        self.fc_advantage = nn.Linear(512, 256)
        self.advantage = nn.Linear(256, n_actions)

    def forward(self, state):
        state = state.long()
        x = self.embedding(state).view(state.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)

        val = F.relu(self.fc_value(x))
        val = self.value(val)

        adv = F.relu(self.fc_advantage(x))
        adv = self.advantage(adv)

        q_values = val + (adv - adv.mean(dim=1, keepdim=True))
        return q_values

class AgentDDQN:
    def __init__(self, input_dims, n_actions, learning_rate=0.005, batch_size=1024,
                 epsilon_decay=0.995, gamma=0.9):
        self.current_model = DuelingQNetwork(input_dims, n_actions)
        self.target_model = DuelingQNetwork(input_dims, n_actions)
        self.target_model.load_state_dict(self.current_model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.current_model.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(100000)
        self.epsilon = 1.0
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.losses = []

    def select_action(self, state, current_direction):
        state = torch.FloatTensor(state).view(1, -1)  # Flatten the state
        with torch.no_grad():
            q_values = self.current_model(state)

        valid_actions = self.get_valid_actions(current_direction)
        valid_q_values = q_values[0, valid_actions]  # Get Q-values only for valid actions

        if random.random() > self.epsilon:
            action_index = valid_q_values.argmax().item()  # Choose best action based on current policy
            action = valid_actions[action_index]
        else:
            action = random.choice(valid_actions)  # Choose a random valid action

        # Print the Q-values and the chosen action for debugging purposes
        # print("Q-values:", q_values.cpu().numpy())
        # print("Selected action:", action)

        return action


    def train(self, envs, render=False, batch_size=16):
        total_reward = 0

        def run_env(idx, env):
            state = env.reset()
            current_direction = env.direction
            done = False
            reward_sum = 0

            while not done:
                action = self.select_action(state, current_direction)
                next_state, reward, done, _ = env.step(action)
                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                current_direction = env.direction
                reward_sum += reward
                if render:
                    env.render()
                    pygame.time.wait(100)
            return reward_sum

        for i in range(0, len(envs), batch_size):
            batch_envs = envs[i:i+batch_size]
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = [executor.submit(run_env, idx, env) for idx, env in enumerate(batch_envs)]
                for future in as_completed(futures):
                    total_reward += future.result()

        # Training after all episodes are done
        if len(self.replay_buffer) >= self.batch_size:
            mini_batch_number = 64
            print("Updating model")
            minibatch_size = len(self.replay_buffer) // mini_batch_number  # Define minibatch size as 1/n th of replay buffer size

            # Ensure the minibatch size is at least 1 and not smaller than the regular batch size
            minibatch_size = max(minibatch_size, 1)
            minibatch_size = min(minibatch_size, self.batch_size)

            for _ in range(mini_batch_number):  # Perform 16 updates to cover the entire replay buffer approximately
                batch = self.replay_buffer.sample(minibatch_size)
                loss = self.compute_loss(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.losses.append(loss.item())
            self.replay_buffer.clear()

        return total_reward

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch

        states = torch.FloatTensor(states).view(states.shape[0], -1)
        next_states = torch.FloatTensor(next_states).view(next_states.shape[0], -1)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        q_values = self.current_model(states)
        next_q_values = self.current_model(next_states)
        next_q_state_values = self.target_model(next_states)

        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = rewards + self.gamma * next_q_value * (1 - dones)

        loss = (q_value - expected_q_value.detach()).pow(2).mean()
        return loss

    def update_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_decay

    def update_target_network(self):
        self.target_model.load_state_dict(self.current_model.state_dict())

    def get_valid_actions(self, current_direction):
        direction_map = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
        opposite_direction_map = {0: 1, 1: 0, 2: 3, 3: 2}
        current_direction_index = direction_map.index(current_direction)
        valid_actions = [i for i in range(len(direction_map)) if i != opposite_direction_map[current_direction_index]]
        return valid_actions

    def save_model(self, filepath):
        state = {
            'model_state_dict': self.current_model.state_dict(),
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'epsilon_decay': self.epsilon_decay,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate
        }
        torch.save(state, filepath)

    def load_model(self, filepath):
        state = torch.load(filepath)
        self.current_model.load_state_dict(state['model_state_dict'])
        self.target_model.load_state_dict(self.current_model.state_dict())
        self.epsilon = state['epsilon']
        self.gamma = state['gamma']
        self.epsilon_decay = state['epsilon_decay']
        self.batch_size = state['batch_size']
        self.learning_rate = state['learning_rate']
        self.optimizer = optim.Adam(self.current_model.parameters(), lr=self.learning_rate)

def play_with_model(model, env):
    state = env.reset()
    done = False

    while not done:
        state = torch.FloatTensor(state).view(1, -1)  # Flatten the state
        action = model(state).argmax(1).item()
        next_state, reward, done, _ = env.step(action)
        state = next_state
        env.render()
        pygame.time.wait(100)

    env.close()
    print(f"Game Over! Score: {env.score}")

if __name__ == "__main__":
    num_episodes = 100  # 1000
    workers = 1  # 256
    envs = []
    size = 20
    obstacle_number = 15
    # obstacles = [(5, 5)] + [(15, 15)] + [(5, 17)]

    for _ in range(workers):
        random_number = random.randint(0, size - 2)
        random_number_2 = random.randint(0, size - 2)
        obstacles = [(random_number, random_number_2), (random_number + 1, random_number_2),
                     (random_number, random_number_2 + 1), (random_number + 1, random_number_2 + 1)] + \
                    [(random.randint(0, size), random.randint(0, size)) for _ in
                     range(random.randint(0, obstacle_number))]
        env = SnakeGameAI(obstacles=obstacles, enemy_count=random.randint(1, 1), apple_count=random.randint(1, 5),
                          headless=True, size=size)
        envs.append(env)

    observation = envs[0].reset()
    input_dims = np.prod(observation.shape)  # Adjusting for flattened input
    n_actions = envs[0].action_space.n
    agent = AgentDDQN(input_dims, n_actions, batch_size=workers*64, learning_rate=0.00025, epsilon_decay=0.005, gamma=0.9)

    rewards = []

    import os
    # Create a directory to store logs for debugging
    log_dir = r'C:\Users\jmask\OneDrive\Pulpit\snake'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    for episode in tqdm(range(num_episodes)):
        envs = []
        for _ in range(workers):
            random_number = random.randint(0, size - 2)
            random_number_2 = random.randint(0, size - 2)
            obstacles = [(random_number, random_number_2), (random_number + 1, random_number_2),
                         (random_number, random_number_2 + 1), (random_number + 1, random_number_2 + 1)] + \
                        [(random.randint(0, size), random.randint(0, size)) for _ in
                         range(random.randint(0, obstacle_number))]
            env = SnakeGameAI(obstacles=obstacles, enemy_count=random.randint(1, 2), apple_count=random.randint(1, 5),
                              headless=True, size=size)
            envs.append(env)

        reward = agent.train(envs, render=False)
        # reward = agent.train_with_logging(envs, episode, render=True, log_dir=log_dir)
        rewards.append(reward)
        print(f"Episode {episode}, Reward: {reward}")

        if episode % 10 == 0:
            agent.update_target_network()

        if episode % 100 == 0:
            agent.save_model(os.path.join(log_dir, f'model_episode_{episode}.pth'))

        agent.update_epsilon()

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode over Training')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(agent.losses, label='Loss per learning step')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss per learning step over Training')
    plt.legend()
    plt.grid(True)
    plt.show()

    random_number = random.randint(0, size - 2)
    random_number_2 = random.randint(0, size - 2)
    obstacles = [(random_number, random_number_2), (random_number + 1, random_number_2),
                 (random_number, random_number_2 + 1), (random_number + 1, random_number_2 + 1)] + \
                [(random.randint(0, size), random.randint(0, size)) for _ in range(random.randint(0, obstacle_number))]

    pygame.init()
    env_to_play = SnakeGameAI(obstacles=obstacles, enemy_count=1, apple_count=3, headless=False, size=size)
    play_with_model(agent.current_model, env_to_play)
