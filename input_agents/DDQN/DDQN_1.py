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


class DuelingQNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, dropout_rate=1 / 5):
        super(DuelingQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc_value = nn.Linear(128, 64)
        self.value = nn.Linear(64, 1)
        self.fc_advantage = nn.Linear(128, 64)
        self.advantage = nn.Linear(64, n_actions)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)

        val = torch.relu(self.fc_value(x))
        val = self.value(val)

        adv = torch.relu(self.fc_advantage(x))
        adv = self.advantage(adv)

        q_values = val + (adv - adv.mean(dim=1, keepdim=True))
        return q_values


class AgentDDQN:
    def __init__(self, input_dims, n_actions, learning_rate=0.001, batch_size=64, epsilon_decay=0.001, gamma=0.9):
        self.current_model = DuelingQNetwork(input_dims, n_actions)
        self.target_model = DuelingQNetwork(input_dims, n_actions)
        self.target_model.load_state_dict(self.current_model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.current_model.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(100000)
        self.epsilon = 1.0
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.losses = []

    def select_action(self, state, env):
        if random.random() > self.epsilon:
            state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                action = self.current_model(state).argmax(1).item()
        else:
            action = random.randint(0, env.action_space.n - 1)
        return action

    def train(self, env, num_episodes=1000, max_steps=1000):
        rewards_history = []
        for episode in tqdm(range(num_episodes), desc="Training Progress"):
            state = self.extract_state_from_env(env)
            done = False
            total_reward = 0

            for step in range(max_steps):
                # Select an action
                action = self.select_action(state, env)

                # Perform the action in the environment
                next_state, reward, done, _ = env.step(action)
                next_state = self.extract_state_from_env(env)

                # Store the experience in the replay buffer
                self.replay_buffer.push(state, action, reward, next_state, done)

                # Update state
                state = next_state
                total_reward += reward

                # Perform learning
                if len(self.replay_buffer) >= self.batch_size:
                    self.learn()

                if done:
                    break

            # Log and decay epsilon
            rewards_history.append(total_reward)
            self.update_epsilon()

            # Update target network periodically
            if episode % 10 == 0:
                self.update_target_network()

        # Plot the training curve
        self.plot_training_curve(rewards_history)

    def learn(self):
        # Sample a batch from the replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        loss = self.compute_loss(batch)

        # Perform backpropagation
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
        self.epsilon = max(self.epsilon - self.epsilon_decay, 0.00)

    def update_target_network(self):
        self.target_model.load_state_dict(self.current_model.state_dict())

    def extract_state_from_env(self, env):
        """
        This function extracts the state for the agent in the form of enhanced information:
        4 directions for snake's movement, 3 danger values (obstacle type and distance),
        and food direction (nearest apple).
        """
        head = env.snake[0]
        directions = [
            (-1, 0),  # Up
            (1, 0),   # Down
            (0, -1),  # Left
            (0, 1),   # Right
        ]

        # Get danger info in straight, left, and right directions
        danger_straight = self.get_danger_in_direction(env, env.direction)
        danger_left = self.get_danger_in_direction(env, self.turn_left(env.direction))
        danger_right = self.get_danger_in_direction(env, self.turn_right(env.direction))

        # Get nearest apple's direction
        food_direction = self.get_food_direction(env)

        # Construct state
        state = [
            # Direction
            env.direction == (-1, 0),  # Up
            env.direction == (1, 0),   # Down
            env.direction == (0, -1),  # Left
            env.direction == (0, 1),   # Right

            # Danger (type and distance)
            danger_straight[0], danger_straight[1],  # What is in front and how far
            danger_left[0], danger_left[1],          # What is on the left and how far
            danger_right[0], danger_right[1],        # What is on the right and how far

            # Food direction (x, y)
            food_direction[0],  # Distance to nearest food on x-axis
            food_direction[1]   # Distance to nearest food on y-axis
        ]

        return np.array(state, dtype=float)

    def get_danger_in_direction(self, game, direction):
        """ Returns the type of obstacle (1 = wall, 2 = snake) and distance to the obstacle in the given direction """
        head = game.snake[0]  # SnakeCell object
        distance = 0

        # Access x and y attributes instead of indexing
        next_position = (head.x + direction[0], head.y + direction[1])

        while 0 <= next_position[0] < game.size and 0 <= next_position[1] < game.size:
            distance += 1
            if any(cell.x == next_position[0] and cell.y == next_position[1] for cell in game.snake):
                return 2, distance  # Snake body
            next_position = (next_position[0] + direction[0], next_position[1] + direction[1])

        return 1, distance + 1  # Wall

    def turn_left(self, direction):
        """ Turn left based on the current direction """
        if direction == (-1, 0):
            return (0, -1)  # Up -> Left
        if direction == (1, 0):
            return (0, 1)   # Down -> Right
        if direction == (0, -1):
            return (1, 0)   # Left -> Down
        if direction == (0, 1):
            return (-1, 0)  # Right -> Up

    def turn_right(self, direction):
        """ Turn right based on the current direction """
        if direction == (-1, 0):
            return (0, 1)  # Up -> Right
        if direction == (1, 0):
            return (0, -1)  # Down -> Left
        if direction == (0, -1):
            return (-1, 0)  # Left -> Up
        if direction == (0, 1):
            return (1, 0)   # Right -> Down

    def get_food_direction(self, game):
        """ Finds the closest apple and returns the direction (x, y) to that apple """
        head = game.snake[0]
        closest_apple = min(game.apples, key=lambda apple: abs(apple.x - head.x) + abs(apple.y - head.y))

        # Calculate the distance in x and y directions
        distance_x = closest_apple.x - head.x
        distance_y = closest_apple.y - head.y

        return distance_x, distance_y

    def plot_training_curve(self, rewards_history):
        plt.plot(rewards_history)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Progress')
        plt.show()


def play_with_model(model, env):
    state = env.reset()
    state = agent.extract_state_from_env(env)  # Ensure correct state extraction for 8 values
    done = False

    while not done:
        state = torch.FloatTensor(state).unsqueeze(0)  # Ensure the state is size (1, 8)
        action = model(state).argmax(1).item()
        next_state, reward, done, _ = env.step(action)
        state = agent.extract_state_from_env(env)
        env.render()
        pygame.time.wait(100)

    env.close()
    print(f"Game Over! Score: {env.score}")


if __name__ == '__main__':
    env = SnakeGameAI(obstacles=[], enemy_count=0, apple_count=2, headless=False, size=20)
    agent = AgentDDQN(input_dims=10, n_actions=env.action_space.n)  # input_dims=10 for new state
    agent.train(env, num_episodes=2000)  # Add num_episodes argument

    pygame.init()
    env_to_play = SnakeGameAI(obstacles=[], enemy_count=0, apple_count=2, headless=False, size=20)
    play_with_model(agent.current_model, env_to_play)
