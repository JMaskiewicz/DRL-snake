import pygame
import numpy as np
import random

class SnakeGameAI:
    def __init__(self, model=None, obstacles=None, enemy_snake=None):
        self.snake = None
        self.direction = None
        self.apple = None
        self.score = None
        self.done = None
        self.obstacles = obstacles if obstacles is not None else []

        pygame.init()
        self.size = 64
        self.cell_size = 10
        self.width, self.height = self.size * self.cell_size, self.size * self.cell_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake Game AI')

        self.model = model  # AI model
        self.is_human = True if model is None else False
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.snake = [(self.size // 2, self.size // 2)]
        self.direction = (0, -1)
        self.apple = self.spawn_apple()
        self.score = 0
        self.done = False
        self.obstacles = obstacles if obstacles is not None else []
        return self.get_state()

    def spawn_apple(self):
        while True:
            apple = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if apple not in self.snake and (self.obstacles is None or apple not in self.obstacles):
                return apple

    def get_state(self):
        state = np.zeros((self.size, self.size), dtype=int)
        for s in self.snake:
            state[s] = 1
        state[self.apple] = 2
        for obs in self.obstacles:
            state[obs] = -1
        return state.flatten()

    def step(self, action):
        # Map action to direction
        direction_map = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
        self.direction = direction_map[action]

        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])

        # Check for collisions
        if (new_head[0] < 0 or new_head[0] >= self.size or
            new_head[1] < 0 or new_head[1] >= self.size or
            new_head in self.snake or new_head in self.obstacles):
            self.done = True
            reward = -10  # Punishment for dying
            return self.get_state(), reward, self.done

        self.snake.insert(0, new_head)

        if new_head == self.apple:
            self.score += 1
            reward = 10  # Reward for eating apple
            self.apple = self.spawn_apple()
        else:
            self.snake.pop()
            reward = 0

        return self.get_state(), reward, self.done

    def render(self):
        self.screen.fill((0, 0, 0))
        for x, y in self.snake:
            pygame.draw.rect(self.screen, (0, 255, 0), (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
        pygame.draw.rect(self.screen, (255, 0, 0), (self.apple[0] * self.cell_size, self.apple[1] * self.cell_size, self.cell_size, self.cell_size))
        for x, y in self.obstacles:
            pygame.draw.rect(self.screen, (0, 0, 255), (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
        pygame.display.flip()

    def direction_to_index(self, direction):
        direction_map = {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}
        return direction_map.get(direction, 0)

    def run_game(self):
        while not self.done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.done = True
                    continue
                if event.type == pygame.KEYDOWN and self.is_human:
                    if event.key == pygame.K_w and self.direction != (0, 1):  # Prevent reversing
                        self.direction = (0, -1)
                    elif event.key == pygame.K_s and self.direction != (0, -1):
                        self.direction = (0, 1)
                    elif event.key == pygame.K_a and self.direction != (1, 0):
                        self.direction = (-1, 0)
                    elif event.key == pygame.K_d and self.direction != (-1, 0):
                        self.direction = (1, 0)

            # AI control
            if not self.is_human:
                if self.model:
                    prediction = self.model.predict(self.get_state().reshape(1, -1))
                    self.step(np.argmax(prediction))
                else:
                    print("No AI model provided, switching to manual control.")
                    self.is_human = True

            # Move the snake based on current direction
            if self.is_human:
                self.step(self.direction_to_index(self.direction))

            self.render()
            self.clock.tick(10)  # Control the game speed

        pygame.quit()
        print(f"Game Over! Score: {self.score}")

# Example usage:
if __name__ == '__main__':
    obstacles = [(50, 50), (20, 20), (21, 20), (20, 21), (21, 21)]
    game = SnakeGameAI(obstacles=obstacles)
    game.run_game()
