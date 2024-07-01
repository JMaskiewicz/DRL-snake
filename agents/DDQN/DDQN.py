from game.snake_game import SnakeGameAI

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pygame
from tqdm import tqdm
import matplotlib.pyplot as plt

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, min(len(self.buffer), batch_size))
        state, action, reward, next_state, done = zip(*samples)
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

class DuelingQNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, dropout_rate=1 / 5):
        super(DuelingQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, 1024)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc_value = nn.Linear(512, 512)
        self.value = nn.Linear(512, 1)
        self.fc_advantage = nn.Linear(512, 256)
        self.advantage = nn.Linear(256, n_actions)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)

        val = torch.relu(self.fc_value(x))
        val = self.value(val)

        adv = torch.relu(self.fc_advantage(x))
        adv = self.advantage(adv)

        q_values = val + (adv - adv.mean(dim=1, keepdim=True))
        return q_values

class AgentDDQN:
    def __init__(self, input_dims, n_actions, learning_rate=0.005, env_size=64, input_size=64, batch_size=1024,
                 epsilon_decay=0.995, gamma=0.9):
        self.env_size = env_size
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

    def select_action(self, state, env):
        if random.random() > self.epsilon:
            state = torch.FloatTensor(state).view(1, -1)  # Flatten the state
            with torch.no_grad():
                action = self.current_model(state).argmax(1).item()
        else:
            action = random.randint(0, env.action_space.n - 1)
        return action

    def train(self, envs):
        states = [env.reset() for env in envs]
        dones = [False] * len(envs)
        total_reward = 0

        while not all(dones):
            actions = [self.select_action(states[idx], envs[idx]) if not dones[idx] else None for idx in range(len(envs))]

            for idx, env in enumerate(envs):
                if not dones[idx]:
                    next_state, reward, done, _ = env.step(actions[idx])
                    self.replay_buffer.push(states[idx], actions[idx], reward, next_state, done)
                    states[idx] = next_state
                    total_reward += reward
                    dones[idx] = done

        if len(self.replay_buffer) >= self.batch_size:
            batch = self.replay_buffer.sample(self.batch_size)
            loss = self.compute_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.losses.append(loss.item())
            print(f"Loss: {loss.item()}")
            self.replay_buffer.clear()

        return total_reward

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch

        states = torch.FloatTensor(states).view(-1, self.env_size * self.env_size)
        next_states = torch.FloatTensor(next_states).view(-1, self.env_size * self.env_size)
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
        self.epsilon = self.epsilon * self.epsilon_decay

    def update_target_network(self):
        self.target_model.load_state_dict(self.current_model.state_dict())

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
    num_episodes = 1000
    workers = 1
    envs = []
    size = 16
    obstacle_number = 0

    for _ in range(workers):
        random_number = random.randint(0, size - 2)
        random_number_2 = random.randint(0, size - 2)
        obstacles = [(random_number, random_number_2), (random_number + 1, random_number_2),
                     (random_number, random_number_2 + 1), (random_number + 1, random_number_2 + 1)] + \
                    [(random.randint(0, size), random.randint(0, size)) for _ in
                     range(random.randint(0, obstacle_number))]
        env = SnakeGameAI(obstacles=[], enemy_count=random.randint(0, 0), apple_count=random.randint(1, 5),
                          headless=True, size=size)
        envs.append(env)

    observation = envs[0].reset()
    input_dims = observation.size
    n_actions = envs[0].action_space.n
    agent = AgentDDQN(input_dims, n_actions, input_size=size, batch_size=128, learning_rate=0.0025, epsilon_decay=0.9975, gamma=0.95)

    rewards = []

    for episode in tqdm(range(num_episodes)):
        reward = agent.train(envs)
        rewards.append(reward)
        print(f"Episode {episode}, Reward: {reward}")

        if episode % 20 == 0:
            agent.update_target_network()

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
    plt.plot(agent.losses, label='Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss per Episode over Training')
    plt.legend()
    plt.grid(True)
    plt.show()

    random_number = random.randint(0, size - 2)
    random_number_2 = random.randint(0, size - 2)
    obstacles = [(random_number, random_number_2), (random_number + 1, random_number_2),
                 (random_number, random_number_2 + 1), (random_number + 1, random_number_2 + 1)] + \
                [(random.randint(0, size), random.randint(0, size)) for _ in range(random.randint(0, obstacle_number))]

    env_to_play = SnakeGameAI(obstacles=[], enemy_count=0, apple_count=2, headless=False, size=size)

    pygame.init()
    env_to_play = SnakeGameAI(obstacles=[], enemy_count=0, apple_count=2, headless=False, size=size)
    play_with_model(agent.current_model, env_to_play)
