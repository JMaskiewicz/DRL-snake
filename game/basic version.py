import pygame
import random
import time

# Initialize pygame
pygame.init()

# Set up the display
size = width, height = 640, 640  # 64x64 grid, each block 10x10 pixels
screen = pygame.display.set_mode(size)
pygame.display.set_caption('Snake Game')

# Colors
black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)  # Color for the obstacles (rocks)

# Grid settings
grid_size = 64
cell_size = 10
clock = pygame.time.Clock()

# Snake init
snake = [(grid_size // 2, grid_size // 2)]
direction = (0, -1)  # Initial direction: up

obstacles = [(50, 50), (20, 20), (21, 20), (20, 21), (21, 21)]  # Example position for a rock

# Apple init
def spawn_apple():
    while True:
        apple = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
        if apple not in snake and apple not in obstacles:
            return apple

apple = spawn_apple()

# Obstacles init
obstacles = [(50, 50), (20, 20), (21, 20), (20, 21), (21, 21)]  # Example position for a rock

def draw_block(color, pos):
    block = pygame.Rect(pos[0]*cell_size, pos[1]*cell_size, cell_size, cell_size)
    pygame.draw.rect(screen, color, block)

def game_over():
    pygame.quit()
    print("Game Over! Your score:", len(snake))
    exit()

# Game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                direction = (0, -1)
            elif event.key == pygame.K_s:
                direction = (0, 1)
            elif event.key == pygame.K_a:
                direction = (-1, 0)
            elif event.key == pygame.K_d:
                direction = (1, 0)

    # Move snake
    new_head = (snake[0][0] + direction[0], snake[0][1] + direction[1])

    # Check game over conditions
    if (new_head[0] < 0 or new_head[0] >= grid_size or new_head[1] < 0 or new_head[1] >= grid_size or
            new_head in snake or new_head in obstacles):
        game_over()

    snake.insert(0, new_head)

    # Check apple eating
    if new_head == apple:
        apple = spawn_apple()
    else:
        snake.pop()

    # Clear screen and draw new frame
    screen.fill(black)
    for block in snake:
        draw_block(green, block)
    draw_block(red, apple)

    for obstacle in obstacles:
        draw_block(blue, obstacle)

    pygame.display.update()
    clock.tick(10)
