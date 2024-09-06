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
    def __init__(self, apple_count=1, headless=True, size=10, grid_range=1):
        super(SnakeGameAI, self).__init__()
        pygame.init()
        self.size = size  # Size of the grid
        self.cell_size = 25  # Size of each cell
        self.border_thickness = 4  # Thickness for the boundary
        self.width, self.height = self.size * self.cell_size, self.size * self.cell_size
        self.header_height = 100  # Additional space for the title and score display
        self.headless = headless
        self.grid_range = grid_range  # This controls how far the grid extends from the head
        self.score = 0

        # Create a display window with extra space for the header
        if not headless:
            self.screen = pygame.display.set_mode((self.width, self.height + self.header_height))
            pygame.display.set_caption('Snake Game AI')

        self.font = pygame.font.SysFont(None, 36)
        self.apple_count = apple_count
        self.apples = []
        self.move_counter = 0

        # Initialize action and observation spaces
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        grid_size = (self.grid_range * 2 + 1) ** 2
        self.observation_space = spaces.Box(low=0, high=2, shape=(6 + grid_size,), dtype=np.float32)

        self.reset()

    def reset(self):
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

    def is_collision(self, point):
        x, y = point
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return True
        if (x, y) in [(cell.x, cell.y) for cell in self.snake]:
            return True
        return False

    def get_state(self):
        head = (self.snake[0].x, self.snake[0].y)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

        # Direction values
        direction_state = [
            self.direction == (-1, 0),  # Up
            self.direction == (1, 0),   # Down
            self.direction == (0, -1),  # Left
            self.direction == (0, 1)    # Right
        ]

        # Apple position relative to the head
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

        # Get the grid relative to the snake's head
        grid_size = 2 * self.grid_range + 1
        grid = np.zeros((grid_size, grid_size), dtype=int)

        for i in range(-self.grid_range, self.grid_range + 1):
            for j in range(-self.grid_range, self.grid_range + 1):
                cell_x = head[0] + i
                cell_y = head[1] + j

                # Check if this position is out of bounds
                if cell_x < 0 or cell_x >= self.size or cell_y < 0 or cell_y >= self.size:
                    grid[i + self.grid_range][j + self.grid_range] = 1  # Mark out-of-bounds as an obstacle
                else:
                    if (cell_x, cell_y) in [(cell.x, cell.y) for cell in self.snake]:
                        grid[i + self.grid_range][j + self.grid_range] = 1
                    elif (cell_x, cell_y) in self.apples:
                        grid[i + self.grid_range][j + self.grid_range] = 2

        grid_flattened = grid.flatten()
        return np.concatenate([direction_state, apple_direction, grid_flattened])

    def render(self, generation=0):
        if self.headless:
            return

        # Fill background
        self.screen.fill((0, 0, 0))

        # Adjust font sizes
        title_font = pygame.font.SysFont(None, 48)  # Larger font for the title
        info_font = pygame.font.SysFont(None, 36)  # Smaller font for the other info

        # Draw title, generation, and score
        title_text = title_font.render("AI LEARNS TO PLAY SNAKE", True, (255, 255, 255))
        model_text = info_font.render("DQN MODEL", True, (255, 255, 255))
        generation_text = info_font.render(f"GENERATION: {generation}", True, (255, 255, 255))
        score_text = info_font.render(f"SCORE: {self.score}", True, (255, 255, 255))

        # Get width of screen for centering the title
        title_rect = title_text.get_rect(center=(self.width // 2, 30))

        # Display header text
        self.screen.blit(title_text, title_rect)  # Centered title
        self.screen.blit(model_text, (20, 70))  # Model text below title
        self.screen.blit(generation_text, (self.width - 240, 70))  # Generation on the right
        self.screen.blit(score_text, (self.width - 240, 110))  # Score below generation

        # Draw snake and apple
        for cell in self.snake:
            color = (0, 255, 0) if cell.is_head else (0, 150, 0)
            pygame.draw.rect(self.screen, color, (
            cell.x * self.cell_size, cell.y * self.cell_size + self.header_height, self.cell_size, self.cell_size))

        for apple in self.apples:
            pygame.draw.rect(self.screen, (255, 0, 0), (
            apple[0] * self.cell_size, apple[1] * self.cell_size + self.header_height, self.cell_size, self.cell_size))

        # Draw game boundary
        pygame.draw.rect(self.screen, (255, 255, 255), (0, self.header_height, self.width, self.height),
                         self.border_thickness)

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

    # clear memory
    def clear(self):
        self.buffer.clear()


class Linear_QNet(nn.Module):
    def __init__(self, input_dims, hidden_dim1, hidden_dim2, n_actions):
        super(Linear_QNet, self).__init__()
        self.fc1 = nn.Linear(input_dims, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, n_actions)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AgentDDQN:
    def __init__(self, input_dims, n_actions, hidden_dim1=512, hidden_dim2=256, learning_rate=0.001, batch_size=256, epsilon_decay=0.0005, gamma=0.9):
        self.current_model = Linear_QNet(input_dims, hidden_dim1, hidden_dim2, n_actions)  # Pass both hidden_dim1 and hidden_dim2
        self.target_model = Linear_QNet(input_dims, hidden_dim1, hidden_dim2, n_actions)   # Pass both hidden_dim1 and hidden_dim2
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
        self.generation = 0

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
            self.generation += 1
            if episode % 1000 == 0 and episode > 0:
                print(f'Episode {episode}/{num_episodes}')
                # clear memory
                self.replay_buffer.clear()

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

            # Append the total reward to history
            rewards_history.append(total_reward)

            # Print the score every 100 episodes
            if episode % 100 == 0:
                print(f'Episode {episode}: Total Reward: {total_reward}')

            self.update_epsilon()

            if episode % 10 == 0:
                self.update_target_network()

        # After training, plot the scores
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
        # Plot the scores over the episodes
        plt.plot(rewards_history)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Progress - Total Rewards Over Episodes')
        plt.show()


def play_with_model(model, env, generations=1):
    state = env.reset()
    done = False

    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)
        action = model(state).argmax(1).item()
        next_state, reward, done, _ = env.step(action)
        state = next_state
        env.render(generation=generations)  # Pass the generation number for display
        pygame.time.wait(100)

    env.close()


if __name__ == '__main__':
    grid_range = 10  # You can change this to control how far the grid looks from the head
    env = SnakeGameAI(headless=False, grid_range=grid_range)

    grid_size = (2 * grid_range + 1) ** 2
    input_dims = 6 + grid_size

    agent = AgentDDQN(input_dims=input_dims, n_actions=env.action_space.n, epsilon_decay=0.0015)
    agent.train(env, num_episodes=100)

    pygame.init()
    env_to_play = SnakeGameAI(headless=False, grid_range=grid_range)
    play_with_model(agent.current_model, env_to_play, generations=agent.generation)
