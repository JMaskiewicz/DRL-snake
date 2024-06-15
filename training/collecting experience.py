"""
DDQN - Collecting Experience

"""

from game.snake_game import SnakeGameAI

import numpy as np
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pygame

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

def compute_loss(batch, current_model, target_model, gamma=0.99):
    states, actions, rewards, next_states, dones = batch
    states = torch.FloatTensor(states)
    next_states = torch.FloatTensor(next_states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    dones = torch.FloatTensor(dones)

    q_values = current_model(states)
    next_q_values = current_model(next_states)
    next_q_state_values = target_model(next_states)

    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = rewards + gamma * next_q_value * (1 - dones)

    loss = (q_value - expected_q_value.detach()).pow(2).mean()
    return loss

def train(envs, current_model, target_model, optimizer, replay_buffer, epsilon, batch_size=1024, gamma=0.99):
    total_reward = 0
    state = [env.reset() for env in envs]
    state = np.stack(state)
    done = [False for _ in envs]

    while not all(done):
        action = []
        for s in state:
            if random.random() > epsilon:
                action.append(current_model(torch.FloatTensor(s).unsqueeze(0)).argmax(1).item())
            else:
                action.append(random.randint(0, envs[0].action_space.n - 1))

        next_state, reward, done, _ = zip(*[env.step(a) for env, a in zip(envs, action)])
        next_state = np.stack(next_state)
        reward = np.stack(reward)
        done = np.stack(done)

        for s, a, r, ns, d in zip(state, action, reward, next_state, done):
            replay_buffer.push(s, a, r, ns, d)

        state = next_state
        total_reward += reward.sum()

        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)
            loss = compute_loss(batch, current_model, target_model, gamma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return total_reward / len(envs)

def play_with_model(model, env):
    state = env.reset()
    done = False

    while not done:
        action = model(torch.FloatTensor(state).unsqueeze(0)).argmax(1).item()
        next_state, reward, done, _ = env.step(action)
        state = next_state
        env.render()
        pygame.time.wait(100)  # Adjust delay to control speed

    env.close()
    print(f"Game Over! Score: {env.score}")

def main():
    num_episodes = 100  # Adjust number of episodes if necessary
    replay_buffer = ReplayBuffer(100000)
    envs = []

    for _ in range(16):  # Train with multiple environments
        random_number = random.randint(0, 62)
        random_number_2 = random.randint(0, 62)
        obstacles = [(random_number, random_number_2), (random_number + 1, random_number_2),
                     (random_number, random_number_2 + 1), (random_number + 1, random_number_2 + 1)] + \
                    [(random.randint(0, 63), random.randint(0, 63)) for _ in range(random.randint(0, 20))]
        env = SnakeGameAI(obstacles=obstacles, enemy_count=random.randint(0, 3), apple_count=random.randint(1, 3),
                          headless=True)
        envs.append(env)

    current_model = DQN(envs[0].observation_space.shape[0], envs[0].action_space.n)
    target_model = DQN(envs[0].observation_space.shape[0], envs[0].action_space.n)
    optimizer = optim.Adam(current_model.parameters())

    update_target(current_model, target_model)

    epsilon = 1.0  # Initial epsilon value
    epsilon_decay = 0.1  # Decay rate per episode

    for episode in range(num_episodes):
        envs = []

        for _ in range(16):  # Train with multiple environments
            random_number = random.randint(0, 62)
            random_number_2 = random.randint(0, 62)
            obstacles = [(random_number, random_number_2), (random_number + 1, random_number_2),
                         (random_number, random_number_2 + 1), (random_number + 1, random_number_2 + 1)] + \
                        [(random.randint(0, 63), random.randint(0, 63)) for _ in range(random.randint(0, 20))]
            env = SnakeGameAI(obstacles=obstacles, enemy_count=random.randint(0, 3), apple_count=random.randint(1, 3),
                              headless=True)
            envs.append(env)

        reward = train(envs, current_model, target_model, optimizer, replay_buffer, epsilon)
        print(f"Episode {episode}, Reward: {reward}")

        if episode % 10 == 0:
            update_target(current_model, target_model)

        epsilon = max(0.0, epsilon - epsilon_decay)

    random_number = random.randint(0, 62)
    random_number_2 = random.randint(0, 62)
    obstacles = [(random_number, random_number_2), (random_number+1, random_number_2), (random_number, random_number_2+1), (random_number+1, random_number_2+1)]+\
    [(random.randint(0, 63), random.randint(0, 63)) for _ in range(random.randint(0, 20))]
    env_to_play = SnakeGameAI(obstacles=obstacles, enemy_count=2, apple_count=2, headless=False)

    # After training, play the game with the trained model
    play_with_model(current_model, env_to_play)

if __name__ == "__main__":
    main()









