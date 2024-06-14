import pygame
import numpy as np
import random


class SnakeGameAI:
    def __init__(self, model=None, obstacles=None, enemy_count=1, apple_count=1):
        pygame.init()
        self.size = 64
        self.cell_size = 10
        self.width, self.height = self.size * self.cell_size, self.size * self.cell_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake Game AI')

        self.model = model  # AI model
        self.is_human = True if model is None else False
        self.clock = pygame.time.Clock()
        self.obstacles = obstacles if obstacles is not None else []

        self.enemy_count = enemy_count  # Number of enemy snakes
        self.apple_count = apple_count  # Number of apples to spawn
        self.enemies = []  # Initialize enemies here
        self.apples = []  # Store apple positions

        self.reset()

    def reset(self):
        self.snake = [(self.size // 2, self.size // 2)]
        self.direction = (0, -1)
        self.apples = [self.spawn_apple() for _ in range(self.apple_count)]  # Spawn multiple apples
        self.score = 0
        self.done = False
        self.enemies = [self.spawn_enemy() for _ in range(self.enemy_count)]
        return self.get_state()

    def spawn_apple(self):
        while True:
            apple = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if apple not in self.snake and all(
                    apple not in enemy for enemy in self.enemies) and apple not in self.obstacles:
                return apple

    def spawn_enemy(self):
        enemy = []
        start_pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
        for i in range(5):  # Length of enemy snake
            enemy.append((start_pos[0], start_pos[1] + i))
        return enemy

    def move_enemy(self, enemy):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        direction = random.choice(directions)
        new_head = (enemy[0][0] + direction[0], enemy[0][1] + direction[1])

        if not (0 <= new_head[0] < self.size and 0 <= new_head[1] < self.size) or new_head in enemy:
            direction = (-direction[0], -direction[1])  # Reverse direction
            new_head = (enemy[0][0] + direction[0], enemy[0][1] + direction[1])

        enemy.insert(0, new_head)
        if new_head == self.apple:
            self.apple = self.spawn_apple()  # Respawn apple
        else:
            enemy.pop()

    def get_state(self):
        state = np.zeros((self.size, self.size), dtype=int)
        for s in self.snake:
            state[s] = 1
        for apple in self.apples:  # Multiple apples
            state[apple] = 2
        for obs in self.obstacles:
            state[obs] = -1
        for enemy in self.enemies:
            for part in enemy:
                state[part] = -2
        return state.flatten()

    def step(self, action):
        direction_map = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        self.direction = direction_map[action]

        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])

        collision_objects = self.snake + self.obstacles + [part for enemy in self.enemies for part in enemy]
        if (new_head[0] < 0 or new_head[0] >= self.size or
                new_head[1] < 0 or new_head[1] >= self.size or
                new_head in collision_objects):
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

        for enemy in self.enemies:
            self.move_enemy(enemy)

        return self.get_state(), reward, self.done

    def direction_to_index(self, direction):
        # Convert direction tuple to index for action
        direction_map = {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}
        return direction_map.get(direction, 0)


    def render(self):
        self.screen.fill((0, 0, 0))
        for x, y in self.snake:
            pygame.draw.rect(self.screen, (0, 255, 0),
                             (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
        pygame.draw.rect(self.screen, (255, 0, 0), (
        self.apple[0] * self.cell_size, self.apple[1] * self.cell_size, self.cell_size, self.cell_size))
        for enemy in self.enemies:
            for x, y in enemy:
                pygame.draw.rect(self.screen, (255, 200, 0),
                                 (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
        for x, y in self.obstacles:
            pygame.draw.rect(self.screen, (0, 0, 255),
                             (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
        pygame.display.flip()

    def run_game(self):
        while not self.done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.done = True
                    continue
                if event.type == pygame.KEYDOWN and self.is_human:
                    if event.key == pygame.K_w and self.direction != (0, 1):
                        self.direction = (0, -1)
                    elif event.key == pygame.K_s and self.direction != (0, -1):
                        self.direction = (0, 1)
                    elif event.key == pygame.K_a and self.direction != (1, 0):
                        self.direction = (-1, 0)
                    elif event.key == pygame.K_d and self.direction != (-1, 0):
                        self.direction = (1, 0)

            if not self.is_human:
                if self.model:
                    prediction = self.model.predict(self.get_state().reshape(1, -1))
                    self.step(np.argmax(prediction))
                else:
                    print("No AI model provided, switching to manual control.")
                    self.is_human = True

            if self.is_human:
                self.step(self.direction_to_index(self.direction))

            self.render()
            self.clock.tick(10)

        pygame.quit()
        print(f"Game Over! Score: {self.score}")


# Example usage:
if __name__ == '__main__':
    obstacles = [(50, 50), (20, 20), (21, 20), (20, 21), (21, 21)]
    game = SnakeGameAI(obstacles=obstacles, enemy_count=2, apple_count=2)
    game.run_game()
