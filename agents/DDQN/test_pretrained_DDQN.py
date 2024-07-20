from game.snake_game import SnakeGameAI


import os
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
        self.epsilon = 1.0
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

    def select_action(self, state, current_direction):
        state = torch.FloatTensor(state).view(1, -1)  # Flatten the state
        with torch.no_grad():
            q_values = self.current_model(state)

        valid_actions = self.get_valid_actions(current_direction)
        valid_q_values = q_values[0, valid_actions]  # Get Q-values only for valid actions

        action_index = valid_q_values.argmax().item()  # Choose best action based on current policy
        action = valid_actions[action_index]

        return action

    def get_valid_actions(self, current_direction):
        direction_map = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
        opposite_direction_map = {0: 1, 1: 0, 2: 3, 3: 2}
        current_direction_index = direction_map.index(current_direction)
        valid_actions = [i for i in range(len(direction_map)) if i != opposite_direction_map[current_direction_index]]
        return valid_actions

    def load_model(self, filepath):
        self.current_model.load_state_dict(torch.load(filepath))
        self.target_model.load_state_dict(self.current_model.state_dict())


def play_with_model(agent, env):
    state = env.reset()
    done = False

    while not done:
        current_direction = env.direction
        action = agent.select_action(state, current_direction)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        env.render()
        pygame.time.wait(100)

    env.close()
    print(f"Game Over! Score: {env.score}")


if __name__ == "__main__":
    size = 20
    obstacle_number = 10
    random_number = random.randint(0, size - 2)
    random_number_2 = random.randint(0, size - 2)
    obstacles = [(random_number, random_number_2), (random_number + 1, random_number_2),
                 (random_number, random_number_2 + 1), (random_number + 1, random_number_2 + 1)] + \
                [(random.randint(0, size), random.randint(0, size)) for _ in range(random.randint(0, obstacle_number))]

    env = SnakeGameAI(obstacles=obstacles, enemy_count=1, apple_count=3, headless=False, size=size)

    observation = env.reset()
    input_dims = np.prod(observation.shape)  # Adjusting for flattened input
    n_actions = env.action_space.n
    agent = AgentDDQN(input_dims, n_actions, batch_size=64, learning_rate=0.00025, epsilon_decay=0.005, gamma=0.9)

    pretrained_model_path = r'C:\Users\jmask\OneDrive\Pulpit\snake\model_episode_0.pth'
    if os.path.exists(pretrained_model_path):
        agent.load_model(pretrained_model_path)
        print(f"Loaded pretrained model from {pretrained_model_path}")

    pygame.init()
    play_with_model(agent, env)
