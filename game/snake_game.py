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
        while True:
            start_pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            enemy = [(start_pos[0], start_pos[1] + i) for i in range(5)]
            if all(0 <= part[1] < self.size for part in enemy) and all(part not in self.snake for part in enemy):
                break
        return enemy

    def move_enemy(self, enemy):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        direction = random.choice(directions)
        new_head = (enemy[0][0] + direction[0], enemy[0][1] + direction[1])

        # Check if the new head is out of bounds or collides with itself or obstacles
        if not (0 <= new_head[0] < self.size and 0 <= new_head[
            1] < self.size) or new_head in enemy or new_head in self.obstacles:
            # Choose another valid direction
            valid_directions = [
                (d[0], d[1]) for d in directions
                if 0 <= enemy[0][0] + d[0] < self.size and 0 <= enemy[0][1] + d[1] < self.size
                   and (enemy[0][0] + d[0], enemy[0][1] + d[1]) not in enemy and (
                   enemy[0][0] + d[0], enemy[0][1] + d[1]) not in self.obstacles
            ]
            if valid_directions:
                direction = random.choice(valid_directions)
                new_head = (enemy[0][0] + direction[0], enemy[0][1] + direction[1])
            else:
                # If no valid directions, keep the enemy in place
                return

        enemy.insert(0, new_head)
        if new_head in self.apples:
            self.apples[self.apples.index(new_head)] = self.spawn_apple()  # Respawn apple
        else:
            enemy.pop()

    def get_state(self):
        state = np.zeros((self.size, self.size), dtype=int)
        for s in self.snake:
            if 0 <= s[0] < self.size and 0 <= s[1] < self.size:
                state[s] = 1
        for apple in self.apples:  # Multiple apples
            if 0 <= apple[0] < self.size and 0 <= apple[1] < self.size:
                state[apple] = 2
        for obs in self.obstacles:
            if 0 <= obs[0] < self.size and 0 <= obs[1] < self.size:
                state[obs] = -1
        for enemy in self.enemies:
            for part in enemy:
                if 0 <= part[0] < self.size and 0 <= part[1] < self.size:
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

        # Check if new_head is in any apple positions
        if new_head in self.apples:
            self.score += 1
            reward = 10  # Reward for eating apple
            self.apples[self.apples.index(new_head)] = self.spawn_apple()  # Replace the eaten apple
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
        for i, (x, y) in enumerate(self.snake):
            color = (0, 130, 0) if i == 0 else (0, 255, 0)  # Darker green for the head
            pygame.draw.rect(self.screen, color,
                             (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
        for apple in self.apples:
            pygame.draw.rect(self.screen, (255, 0, 0),
                             (apple[0] * self.cell_size, apple[1] * self.cell_size, self.cell_size, self.cell_size))
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
    random_number = random.randint(0, 62)
    random_number_2 = random.randint(0, 62)
    obstacles = [(random_number, random_number_2), (random_number+1, random_number_2), (random_number, random_number_2+1), (random_number+1, random_number_2+1)]+\
    [(random.randint(0, 63), random.randint(0, 63)) for _ in range(random.randint(0, 20))]
    game = SnakeGameAI(obstacles=obstacles, enemy_count=2, apple_count=2)
    game.run_game()
