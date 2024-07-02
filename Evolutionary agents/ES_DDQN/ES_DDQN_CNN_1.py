"""
DDQN - Collecting Experience

"""

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
    def __init__(self, input_channels, n_actions, dropout_rate=0.2, input_size=20):
        super(DuelingQNetworkCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2)  # Added padding
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)  # Added padding
        self.dropout2 = nn.Dropout(dropout_rate)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # Added padding
        self.dropout3 = nn.Dropout(dropout_rate)
        # Calculate the output size after conv layers to set the input size for fc1 correctly
        conv_output_size = self._get_conv_output((1, 1, 20, 20))  # assuming input size is 20x20
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
        # Helper function to calculate the size of the output from the conv layers
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


class AgentDDQN:
    def __init__(self, input_dims, n_actions, learning_rate=0.00025, env_size=20, input_size=20):
        self.env_size = env_size
        self.current_model = DuelingQNetworkCNN(1, n_actions, input_size=input_size)
        self.target_model = DuelingQNetworkCNN(1, n_actions, input_size=input_size)
        self.target_model.load_state_dict(self.current_model.state_dict())
        self.optimizer = optim.Adam(self.current_model.parameters(), lr=learning_rate)
        self.epsilon = 1.0
        self.gamma = 0.95

    def select_action(self, state, env):
        if random.random() > self.epsilon:
            state = torch.FloatTensor(state).view(1, 1, env.size, env.size)
            action = self.current_model(state).argmax(1).item()
        else:
            action = random.randint(0, env.action_space.n - 1)
        return action

    def train(self, envs, batch_size=2048):
        print('Training...')
        total_reward = 0
        states = [env.reset() for env in envs]
        dones = [False] * len(envs)

        while not all(dones):
            actions = [self.select_action(states[idx], envs[idx]) if not dones[idx] else None for idx in
                       range(len(envs))]
            for idx, env in enumerate(envs):
                if not dones[idx]:
                    next_state, reward, done, _ = env.step(actions[idx])
                    states[idx] = next_state
                    total_reward += reward
                    dones[idx] = done

        return total_reward

    def update_epsilon(self, generation, max_generations):
        self.epsilon = max(0.01, 1.0 - generation / (max_generations / 1.5))

    def update_target_network(self):
        self.target_model.load_state_dict(self.current_model.state_dict())


def evolutionary_algorithm(envs, n_generations=3, population_size=50, elite_size=10, mutation_rate=0.01,
                           target_update_interval=10):
    population = [AgentDDQN(envs[0].observation_space.shape[0], envs[0].action_space.n, input_size=20) for _ in
                  range(population_size)]
    rewards = []

    for generation in tqdm(range(n_generations)):
        fitness = [agent.train(envs) for agent in population]
        average_fitness = np.mean(fitness)
        rewards.append(average_fitness)

        # Print progress
        print(f'Generation {generation}: Average Reward: {average_fitness}')
        best_fitness = max(fitness)
        print(f'Best Reward in Generation {generation}: {best_fitness}')

        elite_indices = np.argsort(fitness)[-elite_size:]
        elites = [population[idx] for idx in elite_indices]

        new_population = elites.copy()
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(elites, 2)
            child = AgentDDQN(envs[0].observation_space.shape[0], envs[0].action_space.n, input_size=20)
            for param1, param2, child_param in zip(parent1.current_model.parameters(),
                                                   parent2.current_model.parameters(),
                                                   child.current_model.parameters()):
                crossover_point = random.randint(0, param1.numel())
                child_param.data[:crossover_point] = param1.data[:crossover_point]
                child_param.data[crossover_point:] = param2.data[crossover_point:]
                if random.random() < mutation_rate:
                    noise = torch.randn_like(child_param) * 0.1
                    child_param.data += noise
            new_population.append(child)

        for agent in new_population:
            agent.update_epsilon(generation, n_generations)

        population = new_population

        # Update target networks periodically
        if generation % target_update_interval == 0:
            for agent in population:
                agent.update_target_network()

    # Return the best agent
    best_agent_idx = np.argmax(fitness)
    best_agent = population[best_agent_idx]
    return best_agent, rewards


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
    workers = 64
    envs = []
    size = 64
    obstacle_number = 0

    for _ in range(workers):
        random_number = random.randint(0, size - 2)
        random_number_2 = random.randint(0, size - 2)
        obstacles = [(random_number, random_number_2), (random_number + 1, random_number_2),
                     (random_number, random_number_2 + 1), (random_number + 1, random_number_2 + 1)] + \
                    [(random.randint(0, size), random.randint(0, size)) for _ in
                     range(random.randint(0, obstacle_number))]
        env = SnakeGameAI(obstacles=obstacles, enemy_count=random.randint(0, 0), apple_count=random.randint(1, 2),
                          headless=True, size=size)
        envs.append(env)

    best_agent, rewards = evolutionary_algorithm(envs, n_generations=3, population_size=50, elite_size=10)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Average Reward per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Average Reward')
    plt.title('Average Reward per Generation over Training')
    plt.legend()
    plt.grid(True)
    plt.show()

    random_number = random.randint(0, size - 2)
    random_number_2 = random.randint(0, size - 2)
    obstacles = [(random_number, random_number_2), (random_number + 1, random_number_2),
                 (random_number, random_number_2 + 1), (random_number + 1, random_number_2 + 1)] + \
                [(random.randint(0, size), random.randint(0, size)) for _ in range(random.randint(0, obstacle_number))]
    env_to_play = SnakeGameAI(obstacles=obstacles, enemy_count=1, apple_count=2, headless=False, size=size)

    # After training, play the game with the best model
    pygame.init()
    env_to_play = SnakeGameAI(obstacles=obstacles, enemy_count=1, apple_count=2, headless=False, size=size)
    play_with_model(best_agent.current_model, env_to_play)



