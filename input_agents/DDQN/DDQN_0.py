import pygame
import gym
from gym import spaces
from tqdm import tqdm
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt


class SnakeCell:
    """Represents a single cell of the snake."""

    def __init__(self, x, y, is_head=False):
        self.x = x
        self.y = y
        self.is_head = is_head


class SnakeGameAI(gym.Env):
    def __init__(self, apple_count=1, headless=True, size=20):
        super(SnakeGameAI, self).__init__()
        pygame.init()
        self.size = size  # Size of the grid
        self.cell_size = 25  # Size of each cell
        self.width, self.height = self.size * self.cell_size, self.size * self.cell_size
        self.headless = headless
        self.score = 0

        # Create a display window
        if not headless:
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('Snake Game AI')

        self.font = pygame.font.SysFont(None, 36)
        self.apple_count = apple_count
        self.apples = []
        self.move_counter = 0

        # Initialize action and observation spaces
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(low=0, high=1, shape=(9,), dtype=np.float32)

        self.reset()

    def reset(self):
        # Snake as a list of SnakeCell objects
        self.snake = [SnakeCell(self.size // 2, self.size // 2, is_head=True),
                      SnakeCell(self.size // 2, self.size // 2 + 1),
                      SnakeCell(self.size // 2, self.size // 2 + 2)]
        self.direction = (0, -1)  # Start moving up
        self.apples = [self.spawn_apple() for _ in range(self.apple_count)]
        self.score = 0
        self.done = False
        self.move_counter = 0
        return self.get_state()

    def spawn_apple(self):
        while True:
            apple = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if apple not in [(cell.x, cell.y) for cell in self.snake]:
                return apple

    def step(self, action):
        direction_map = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
        self.direction = direction_map[action]
        new_head = (self.snake[0].x + self.direction[0], self.snake[0].y + self.direction[1])

        if self.is_collision(new_head):
            self.done = True
            return self.get_state(), -10, True, {}

        # Move snake
        self.snake.insert(0, SnakeCell(new_head[0], new_head[1], is_head=True))
        self.snake[1].is_head = False

        if new_head in self.apples:
            self.score += 1
            self.apples[self.apples.index(new_head)] = self.spawn_apple()
            reward = 10
        else:
            self.snake.pop()  # Remove the tail
            reward = -0.01  # Small penalty for each step

        self.move_counter += 1
        if self.move_counter > 100 * len(self.snake):
            self.done = True
            reward = -10

        return self.get_state(), reward, self.done, {}

    def get_state(self):
        """
        Simplified state:
        - 4 values for direction (Up, Down, Left, Right)
        - 3 values for danger (straight, left, right)
        - 2 values for the relative position of the apple
        """
        head = (self.snake[0].x, self.snake[0].y)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

        danger_straight = self.is_collision((head[0] + self.direction[0], head[1] + self.direction[1]))
        danger_left = self.is_collision((head[0] - self.direction[1], head[1] + self.direction[0]))
        danger_right = self.is_collision((head[0] + self.direction[1], head[1] - self.direction[0]))

        apple_direction = [0, 0]
        apple = self.apples[0]
        if apple[0] < head[0]:
            apple_direction[0] = -1
        elif apple[0] > head[0]:
            apple_direction[0] = 1
        if apple[1] < head[1]:
            apple_direction[1] = -1
        elif apple[1] > head[1]:
            apple_direction[1] = 1

        return np.array([
            self.direction == (-1, 0),  # Up
            self.direction == (1, 0),   # Down
            self.direction == (0, -1),  # Left
            self.direction == (0, 1),   # Right
            danger_straight,
            danger_left,
            danger_right,
            apple_direction[0],         # x direction (-1 for left, 1 for right)
            apple_direction[1]          # y direction (-1 for up, 1 for down)
        ], dtype=float)

    def is_collision(self, point):
        x, y = point
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return True
        if (x, y) in [(cell.x, cell.y) for cell in self.snake]:
            return True
        return False

    def render(self):
        if self.headless:
            return
        self.screen.fill((0, 0, 0))

        for cell in self.snake:
            color = (0, 255, 0) if cell.is_head else (0, 150, 0)
            pygame.draw.rect(self.screen, color, (cell.x * self.cell_size, cell.y * self.cell_size, self.cell_size, self.cell_size))

        for apple in self.apples:
            pygame.draw.rect(self.screen, (255, 0, 0), (apple[0] * self.cell_size, apple[1] * self.cell_size, self.cell_size, self.cell_size))

        pygame.display.flip()

    def close(self):
        pygame.quit()

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, min(len(self.buffer), batch_size))
        state, action, reward, next_state, done = zip(*samples)
        return np.stack(state), action, reward, np.stack(next_state), done

    def __len__(self):
        return len(self.buffer)


class Linear_QNet(nn.Module):
    def __init__(self, input_dims, hidden_dim, n_actions):
        super(Linear_QNet, self).__init__()
        self.fc1 = nn.Linear(input_dims, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = self.fc2(x)
        return x


class AgentDDQN:
    def __init__(self, input_dims, n_actions, hidden_dim=256, learning_rate=0.001, batch_size=64, epsilon_decay=0.001, gamma=0.9):
        self.current_model = Linear_QNet(input_dims, hidden_dim, n_actions)
        self.target_model = Linear_QNet(input_dims, hidden_dim, n_actions)
        self.target_model.load_state_dict(self.current_model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.current_model.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(100000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.losses = []
        self.reward_for_apple = 10.0
        self.penalty_for_collision = -10.0
        self.step_penalty = -0.01

    def select_action(self, state, env):
        if random.random() > self.epsilon:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = self.current_model(state).argmax(1).item()
        else:
            action = random.randint(0, env.action_space.n - 1)
        return action

    def update_target_network(self):
        self.target_model.load_state_dict(self.current_model.state_dict())

    def train(self, env, num_episodes=1000, max_steps=1000):
        rewards_history = []
        for episode in tqdm(range(num_episodes)):
            state = env.reset()
            done = False
            total_reward = 0

            for step in range(max_steps):
                action = self.select_action(state, env)
                next_state, reward, done, _ = env.step(action)
                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if len(self.replay_buffer) >= self.batch_size:
                    self.learn()

                if done:
                    break

            rewards_history.append(total_reward)
            self.update_epsilon()

            if episode % 10 == 0:
                self.update_target_network()

        self.plot_training_curve(rewards_history)

    def learn(self):
        batch = self.replay_buffer.sample(self.batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
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
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

    def plot_training_curve(self, rewards_history):
        plt.plot(rewards_history)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Progress')
        plt.show()


def play_with_model(model, env):
    state = env.reset()
    done = False

    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)
        action = model(state).argmax(1).item()
        next_state, reward, done, _ = env.step(action)
        state = next_state
        env.render()
        pygame.time.wait(100)

    env.close()


if __name__ == '__main__':
    env = SnakeGameAI(headless=False)
    agent = AgentDDQN(input_dims=9, n_actions=env.action_space.n)
    agent.train(env, num_episodes=1000)

    pygame.init()
    env_to_play = SnakeGameAI(headless=False)
    play_with_model(agent.current_model, env_to_play)