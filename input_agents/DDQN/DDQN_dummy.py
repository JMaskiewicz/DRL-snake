import pygame
import gym
from gym import spaces
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
        # Action logic: 0=up, 1=down, 2=left, 3=right
        direction_map = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        self.direction = direction_map[action]
        new_head = (self.snake[0].x + self.direction[0], self.snake[0].y + self.direction[1])

        # Debugging: Print snake head position and new direction
        print(f"Snake head: {self.snake[0].x, self.snake[0].y}, New head: {new_head}")

        # Check for wall collisions first
        if new_head[0] < 0 or new_head[0] >= self.cols or new_head[1] < 0 or new_head[1] >= self.rows:
            print(f"Collision with wall at: {new_head}")
            self.done = True
            return self.get_state(), -10, True, {}

        # Check if apple is eaten
        if new_head in self.apples:
            self.score += 1
            self.apples[self.apples.index(new_head)] = self.spawn_apple()
            reward = 20
            self.move_counter = 0  # Reset move counter on apple consumption
        else:
            # Remove the snake's tail only if no apple is eaten
            tail = self.snake.pop()

        # Insert the new head and update the snake body
        self.snake.insert(0, SnakeCell(new_head[0], new_head[1], is_head=True))
        self.snake[1].is_head = False

        # After moving, check for self-collision
        if (new_head[0], new_head[1]) in [(cell.x, cell.y) for cell in self.snake[1:]]:
            print(f"Collision with self at: {new_head}")
            self.done = True
            return self.get_state(), -10, True, {}

        # Debugging: Print move counter and game status
        self.move_counter += 1
        print(f"Move counter: {self.move_counter}, Done: {self.done}")

        # Add a higher limit for move counter to ensure it doesn't end prematurely
        if self.move_counter > 1000:
            self.done = True
            print("Game over: Max moves exceeded!")
            reward = -10

        return self.get_state(), reward, self.done, {}

    def is_collision(self, point):
        x, y = point
        # Check for wall collisions
        if x < 0 or x >= self.cols or y < 0 or y >= self.rows:
            print(f"Collision with wall at: {point}")
            return True
        # Check for self-collisions
        if (x, y) in [(cell.x, cell.y) for cell in self.snake]:
            print(f"Collision with self at: {point}")
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


# Main game logic for testing play
def play_with_model(agent, env, generations=1):
    """Play with the current model and allow exploration using epsilon."""
    state = env.reset()
    done = False

    while not done:
        action = agent.select_action(state, env)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        env.render(generation=generations)

        pygame.time.wait(200)  # Adjust wait time for snake moves

    env.close()


# Run the main loop for testing the snake environment
if __name__ == '__main__':
    pygame.init()
    grid_range = 1
    env = SnakeGameAI(headless=False, grid_range=grid_range)

    # Dummy agent for testing (this can be replaced with a trained model)
    class DummyAgent:
        def select_action(self, state, env):
            return random.randint(0, env.action_space.n - 1)

    agent = DummyAgent()
    play_with_model(agent, env)

    pygame.quit()
