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


class DuelingQNetworkCNN(nn.Module):
    def __init__(self, input_channels, n_actions, dropout_rate=0.2, input_size=64):
        super(DuelingQNetworkCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=5, stride=2, padding=2)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.dropout3 = nn.Dropout(dropout_rate)
        conv_output_size = self._get_conv_output((1, input_channels, input_size, input_size))
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc_value = nn.Linear(512, 1)
        self.fc_advantage = nn.Linear(512, n_actions)

    def forward(self, state):
        x = torch.relu(self.conv1(state))
        x = self.dropout1(x)
        x = torch.relu(self.conv2(x))
        x = self.dropout2(x)
        x = torch.relu(self.conv3(x))
        x = self.dropout3(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout4(x)
        val = self.fc_value(x)
        adv = self.fc_advantage(x)
        q_values = val + (adv - adv.mean(dim=1, keepdim=True))
        return q_values

    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.autograd.Variable(torch.rand(*shape))
            output_feat = self._forward_features(input)
            n_size = output_feat.data.view(1, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = torch.relu(self.conv1(x))
        x = self.dropout1(x)
        x = torch.relu(self.conv2(x))
        x = self.dropout2(x)
        x = torch.relu(self.conv3(x))
        x = self.dropout3(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, min(len(self.buffer), batch_size))
        state, action, reward, next_state, done = zip(*samples)
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()


class AgentDDQN:
    def __init__(self, input_dims, n_actions, learning_rate=0.005, env_size=64, input_size=64, batch_size=1024,
                 epsilon_decay=0.995):
        self.env_size = env_size
        self.current_model = DuelingQNetworkCNN(1, n_actions, input_size=input_size)  # input_channels set to 1
        self.target_model = DuelingQNetworkCNN(1, n_actions, input_size=input_size)  # input_channels set to 1
        self.target_model.load_state_dict(self.current_model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.current_model.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(100000)
        self.epsilon = 1.0
        self.gamma = 0.75
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.losses = []

    def select_action(self, state, env):
        if random.random() > self.epsilon:
            state = torch.FloatTensor(state).view(1, 1, env.size, env.size)
            action = self.current_model(state).argmax(1).item()
        else:
            action = random.randint(0, env.action_space.n - 1)
        return action

    def train(self, envs):
        print("Training")
        states = [env.reset() for env in envs]
        dones = [False] * len(envs)
        total_reward = 0

        while not all(dones):
            actions = [self.select_action(states[idx], envs[idx]) if not dones[idx] else None for idx in
                       range(len(envs))]

            for idx, env in enumerate(envs):
                if not dones[idx]:
                    next_state, reward, done, _ = env.step(actions[idx])
                    self.replay_buffer.push(states[idx], actions[idx], reward, next_state, done)
                    states[idx] = next_state
                    total_reward += reward
                    dones[idx] = done

        if len(self.replay_buffer) < self.batch_size:
            print("Not enough samples in the replay buffer to sample a batch.")
            return total_reward

        # learn after each episode
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

        # Debugging print statements to check shapes
        # print(f"States shape before reshape: {states.shape}")
        # print(f"Next_states shape before reshape: {next_states.shape}")

        states = torch.FloatTensor(states).view(-1, 1, self.env_size, self.env_size)
        next_states = torch.FloatTensor(next_states).view(-1, 1, self.env_size, self.env_size)

        # print(f"States shape after reshape: {states.shape}")
        # print(f"Next_states shape after reshape: {next_states.shape}")

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
        state = torch.FloatTensor(state).view(1, 1, env.size, env.size)  # Ensure state is reshaped
        action = model(state).argmax(1).item()
        next_state, reward, done, _ = env.step(action)
        state = next_state
        env.render()
        pygame.time.wait(100)

    env.close()
    print(f"Game Over! Score: {env.score}")


if __name__ == "__main__":
    num_episodes = 100
    workers = 8
    envs = []
    size = 64
    obstacle_number = 5

    for _ in range(workers):
        random_number = random.randint(0, size - 2)
        random_number_2 = random.randint(0, size - 2)
        obstacles = [(random_number, random_number_2), (random_number + 1, random_number_2),
                     (random_number, random_number_2 + 1), (random_number + 1, random_number_2 + 1)] + \
                    [(random.randint(0, size), random.randint(0, size)) for _ in
                     range(random.randint(0, obstacle_number))]
        env = SnakeGameAI(obstacles=obstacles, enemy_count=random.randint(0, 2), apple_count=random.randint(1, 5),
                          headless=True, size=size)
        envs.append(env)

    input_dims = envs[0].observation_space.shape[0]
    n_actions = envs[0].action_space.n
    agent = AgentDDQN(input_dims, n_actions, input_size=size, batch_size=128, learning_rate=0.0025, epsilon_decay=0.99)

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

    env_to_play = SnakeGameAI(obstacles=obstacles, enemy_count=0, apple_count=2, headless=False, size=size)

    # After training, play the game with the trained model
    pygame.init()
    env_to_play = SnakeGameAI(obstacles=obstacles, enemy_count=1, apple_count=2, headless=False, size=size)
    play_with_model(agent.current_model, env_to_play)
