from collections import deque
import numpy as np
import random
import torch

from game import SnakeGameAI, Direction, Point
from model import LinearQNet, QTrainer
from utils import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


def train():
    plot_score = []
    plot_mean_scores = []
    total_score = 0
    best_score = 0

    agent = Agent()
    game = SnakeGameAI()

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move
        reward, game_over, score = game.play_step(final_move)

        # get the new state
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, game_over)

        # remember
        agent.remember(state_old, final_move, reward, state_new, game_over)

        if game_over:
            # train the long memory -- EXPERIENCE REPLAY also plot results
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > best_score:
                best_score = score
                agent.model.save()

            print('GAME #:', agent.n_games, ', Score:', score, ', Best Score:', best_score)
            plot_score.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_score, plot_mean_scores)


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 2023  # parameter to control the randomness
        self.gamma = 0.8  # discount rate < 1 ~ 0.8 or 0.9
        self.memory = deque(maxlen=MAX_MEMORY)  # if we exceed the memory the pop.left() is used

        self.model = LinearQNet(15, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]

        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Moving direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down

            # Bomb location
            False if game.bomb is None else game.bomb.x < game.head.x,  # bomb left
            False if game.bomb is None else game.bomb.x > game.head.x,  # bomb right
            False if game.bomb is None else game.bomb.y < game.head.y,  # bomb up
            False if game.bomb is None else game.bomb.y > game.head.y  # bomb down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))  # pop left if MAX_MEM is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            # random sample
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # random moves: tradeoff b/w EXPLORATION vs EXPLOTATION
        # initially make more random moves but with learned models use the model predictions

        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            # make a random move
            move = random.randint(0, 2)  # 0, 1, 2
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)  # convert the array to tensor
            prediction = self.model(state0)

            move = torch.argmax(prediction).item()  # convert the tensor to a number
            final_move[move] = 1

        return final_move


if __name__ == "__main__":
    train()
