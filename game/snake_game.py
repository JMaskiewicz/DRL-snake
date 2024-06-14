import pygame
import numpy as np
import random

class SnakeGameAI:
    def __init__(self, model=None):
        pygame.init()
        self.size = 64
        self.cell_size = 10
        self.width, self.height = self.size * self.cell_size, self.size * self.cell_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake Game AI')

        self.model = model  # AI model
        self.is_human = True if model is None else False
        self.clock = pygame.time.Clock()

        self.obstacles = [(50, 50)]
        self.reset()

    def reset(self):
        self.snake = [(self.size // 2, self.size // 2)]
        self.direction = (0, -1)  # Initial direction: up
        self.apple = self.spawn_apple()
        self.score = 0
        self.done = False
        self.obstacles = [(50, 50)]  # Static obstacle for simplicity
        return self.get_state()

    def spawn_apple(self):
        while True:
            apple = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if apple not in self.snake and apple not in self.obstacles:
                return apple

    def get_state(self):
        state = np.zeros((self.size, self.size), dtype=int)
        for s in self.snake:
            state[s] = 1  # Snake body part
        state[self.apple] = 2  # Apple location
        for obs in self.obstacles:
            state[obs] = -1  # Obstacles
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

    def run_game(self):
        while not self.done:
            if self.is_human:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.done = True
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_w:
                            self.step(0)
                        elif event.key == pygame.K_s:
                            self.step(1)
                        elif event.key == pygame.K_a:
                            self.step(2)
                        elif event.key == pygame.K_d:
                            self.step(3)
            else:
                # AI makes its move
                prediction = self.model.predict(self.get_state().reshape(1, -1))
                self.step(np.argmax(prediction))

            self.render()
            self.clock.tick(10)

        pygame.quit()
        print(f"Game Over! Score: {self.score}")

# Example usage:
if __name__ == '__main__':
    game = SnakeGameAI()
    game.run_game()
