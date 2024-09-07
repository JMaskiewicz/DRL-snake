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
import os


class SnakeCell:
    """Represents a single cell of the snake."""
    def __init__(self, x, y, is_head=False):
        self.x = x
        self.y = y
        self.is_head = is_head


class SnakeGameAI(gym.Env):
    def __init__(self, apple_count=1, headless=True, rows=30, cols=20, grid_range=1):
        super(SnakeGameAI, self).__init__()
        pygame.init()
        self.rows = rows
        self.cols = cols
        self.cell_size = 30
        self.border_thickness = 5

        self.width, self.height = 720, 1280
        self.grid_width = self.cols * self.cell_size
        self.grid_height = self.rows * self.cell_size

        self.grid_x_offset = (self.width - self.grid_width) // 2
        self.grid_y_offset = (self.height - self.grid_height) // 2 + 50

        self.header_height = 100
        self.headless = headless
        self.grid_range = grid_range
        self.score = 0

        if not headless:
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('Snake Game AI')

        self.font = pygame.font.SysFont(None, 36)
        self.apple_count = apple_count
        self.apples = []
        self.move_counter = 0

        self.action_space = spaces.Discrete(4)
        grid_size = (self.grid_range * 2 + 1) ** 2
        self.observation_space = spaces.Box(low=0, high=2, shape=(6 + grid_size,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.snake = [SnakeCell(self.cols // 2, self.rows // 2, is_head=True),
                      SnakeCell(self.cols // 2, self.rows // 2 + 1),
                      SnakeCell(self.cols // 2, self.rows // 2 + 2)]
        self.direction = (0, -1)
        self.apples = [self.spawn_apple() for _ in range(self.apple_count)]
        self.score = 0
        self.done = False
        self.move_counter = 0
        return self.get_state()

    def spawn_apple(self):
        while True:
            apple = (random.randint(0, self.cols - 1), random.randint(0, self.rows - 1))
            if apple not in [(cell.x, cell.y) for cell in self.snake]:
                return apple

    def step(self, action):
        direction_map = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
        self.direction = direction_map[action]
        new_head = (self.snake[0].x + self.direction[0], self.snake[0].y + self.direction[1])

        if self.is_collision(new_head):
            print("Collision detected, game over!")
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
        if x < 0 or x >= self.cols or y < 0 or y >= self.rows:
            return True
        if (x, y) in [(cell.x, cell.y) for cell in self.snake]:
            return True
        return False

    def get_state(self):
        head = (self.snake[0].x, self.snake[0].y)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        direction_state = [
            self.direction == (-1, 0),
            self.direction == (1, 0),
            self.direction == (0, -1),
            self.direction == (0, 1)
        ]

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

        grid_size = 2 * self.grid_range + 1
        grid = np.zeros((grid_size, grid_size), dtype=int)

        for i in range(-self.grid_range, self.grid_range + 1):
            for j in range(-self.grid_range, self.grid_range + 1):
                cell_x = head[0] + i
                cell_y = head[1] + j
                if cell_x < 0 or cell_x >= self.cols or cell_y < 0 or cell_y >= self.rows:
                    grid[i + self.grid_range][j + self.grid_range] = 1
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

        self.screen.fill((0, 0, 0))

        title_font = pygame.font.SysFont('Verdana', 42, bold=True)
        info_font = pygame.font.SysFont('Arial', 36)

        title_text = title_font.render("AI LEARNS TO PLAY SNAKE", True, (255, 255, 255))
        model_text = info_font.render("DQN MODEL", True, (255, 255, 255))
        generation_text = info_font.render(f"GENERATION: {generation}", True, (255, 255, 255))
        score_text = info_font.render(f"SCORE: {self.score}", True, (255, 255, 255))

        title_rect = title_text.get_rect(center=(self.width // 2, 60))
        model_rect = model_text.get_rect(center=(self.width // 2, 130))
        generation_rect = generation_text.get_rect(center=(self.width // 2, 200))
        score_rect = score_text.get_rect(center=(self.width // 2, 270))

        self.screen.blit(title_text, title_rect)
        self.screen.blit(model_text, model_rect)
        self.screen.blit(generation_text, generation_rect)
        self.screen.blit(score_text, score_rect)

        for cell in self.snake:
            color = (0, 255, 0) if cell.is_head else (0, 150, 0)
            pygame.draw.rect(self.screen, color, (
                self.grid_x_offset + cell.x * self.cell_size,
                self.grid_y_offset + cell.y * self.cell_size,
                self.cell_size,
                self.cell_size))

        for apple in self.apples:
            pygame.draw.rect(self.screen, (255, 0, 0), (
                self.grid_x_offset + apple[0] * self.cell_size,
                self.grid_y_offset + apple[1] * self.cell_size,
                self.cell_size,
                self.cell_size))

        pygame.draw.rect(self.screen, (255, 255, 255),
                         (self.grid_x_offset, self.grid_y_offset, self.grid_width, self.grid_height),
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
    def __init__(self, input_dims, n_actions, hidden_dim1=1024, hidden_dim2=512, learning_rate=0.001, batch_size=256,
                 epsilon_decay=0.0005, gamma=0.9, save_path=None, save_generations=None):
        self.current_model = Linear_QNet(input_dims, hidden_dim1, hidden_dim2, n_actions)
        self.target_model = Linear_QNet(input_dims, hidden_dim1, hidden_dim2, n_actions)
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
        self.generation = 0

        self.save_generations = save_generations
        self.save_directory = save_path
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

    def save_model(self, generation):
        model_path = os.path.join(self.save_directory, f"snake_model_gen_{generation}.pth")
        checkpoint = {
            'model_state_dict': self.current_model.state_dict(),
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'gamma': self.gamma
        }
        torch.save(checkpoint, model_path)
        print(f"Model and hyperparameters saved at generation {generation}")

    def load_model(self, generation, model_path):
        model_path = os.path.join(model_path, f"snake_model_gen_{generation}.pth")
        checkpoint = torch.load(model_path)
        self.current_model.load_state_dict(checkpoint['model_state_dict'])

        self.epsilon = checkpoint['epsilon']
        self.epsilon_decay = checkpoint['epsilon_decay']
        self.gamma = checkpoint['gamma']
        print(f"Model and hyperparameters loaded from generation {generation}")

    def train(self, env, num_episodes=1000, max_steps=1000):
        rewards_history = []
        for episode in tqdm(range(num_episodes)):
            if episode % 1000 == 0 and episode > 0:
                print(f'Episode {episode}/{num_episodes}')
                self.replay_buffer.clear()

            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.select_action(state, env)
                next_state, reward, done, _ = env.step(action)
                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if len(self.replay_buffer) >= self.batch_size:
                    self.learn()

            rewards_history.append(total_reward)

            if episode % 100 == 0:
                print(f'Episode {episode}: Total Reward: {total_reward}')

            self.update_epsilon()

            if episode % 10 == 0:
                self.update_target_network()

            if self.generation in self.save_generations:
                self.save_model(self.generation)

            self.generation += 1

        self.plot_training_curve(rewards_history)

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
        plt.title('Training Progress - Total Rewards Over Episodes')
        plt.show()



def play_with_model(agent, env, generations=1):
    """Play with the current model and allow exploration using epsilon."""
    clock = pygame.time.Clock()  # Create a clock object to control the game's frame rate
    state = env.reset()
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return  # Exit the game if the window is closed

        action = agent.select_action(state, env)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        env.render(generation=generations)

        clock.tick(10)  # Control the frame rate, limiting to 10 frames per second

    pygame.time.wait(2000)  # Wait for a short period after the game ends
    env.close()

def load_model(agent, generation, model_path):
    """Load the saved model and hyperparameters from disk based on the generation number."""
    model_path = os.path.join(model_path, f"snake_model_gen_{generation}.pth")
    checkpoint = torch.load(model_path)
    agent.current_model.load_state_dict(checkpoint['model_state_dict'])

    # Load the epsilon and other hyperparameters as well
    agent.epsilon = checkpoint['epsilon']
    agent.epsilon_decay = checkpoint['epsilon_decay']
    agent.gamma = checkpoint['gamma']
    print(f"Model and hyperparameters loaded from generation {generation}")

def play_with_loaded_model(agent, env, generation, model_path):
    """Play with the loaded model while using epsilon for exploration."""
    load_model(agent, generation, model_path)
    play_with_model(agent, env, generations=generation)

# Main training and loading logic remains the same
if __name__ == '__main__':
    pygame.init()  # Initialize Pygame

    grid_range = 1
    env = SnakeGameAI(headless=False, grid_range=grid_range)

    grid_size = (2 * grid_range + 1) ** 2
    input_dims = 6 + grid_size

    agent = AgentDDQN(input_dims=input_dims, n_actions=env.action_space.n, epsilon_decay=0.001,
                      save_path=r"C:\Pulpit\snake\snake_DQN", save_generations=[0, 100, 500, 1000, 2500, 5000])

    # Train the agent and save models at specified generations
    agent.train(env, num_episodes=1101)

    env_to_play = SnakeGameAI(headless=False, grid_range=grid_range)
    play_with_model(agent, env_to_play, generations=agent.generation)

    # Load and play with the saved modelsprint("Playing with the loaded model")
    print("Playing with the loaded model")
    for gen in [0, 100, 500, 1000]:
        print(f"Playing with model from generation {gen}")
        env_to_play = SnakeGameAI(headless=False, grid_range=grid_range)
        play_with_loaded_model(agent, env_to_play, gen, r"C:\Pulpit\snake\snake_DQN")

    pygame.quit()  # Quit Pygame at the very end
