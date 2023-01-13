from collections import namedtuple
from enum import Enum
import numpy as np
import pygame
import random

pygame.init()
font = pygame.font.SysFont('arial', 25)

# reward for an iteration
# play(action) -> direction
# frame (game iteration)
# is_collision


# 2-D game has 4 dimensions
class Direction(Enum):
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4


# a tuple to determine x and y coordinate
Point = namedtuple('Point', 'x, y')

# defining colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
GREEN = (0, 200, 0)
BLUE = (0, 0, 200)
GRAY = (50, 50, 50)

BLOCK_SIZE = 20
SPEED = 40  # ms


# Agent controlled game
class SnakeGameAI:
    def __init__(self, w=720, h=680):
        self.w = w
        self.h = h

        # initial display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('snake')
        self.clock = pygame.time.Clock()
        self.bomb_time = 100
        self.direction = None
        self.head = None
        self.snake = []
        self.score = None
        self.food, self.bomb = None, None
        self.bomb_tick = None
        self.frame_iteration = None
        self.reset()

    def reset(self):
        # initial game state
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)

        # start with a snake of length 3
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self.bomb = None
        self.bomb_tick = 0
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)

        if self.food in self.snake:
            self._place_food()

    def _place_bomb(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.bomb = Point(x, y)

        if self.bomb in self.snake or self.bomb == self.food:
            self._place_bomb()

    def play_step(self, action):
        self.frame_iteration += 1
        # get the user input -- 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # move the snake -- 2
        self._move(action)
        # changes the head
        self.snake.insert(0, self.head)

        # check if it is game over -- 3
        reward = 0
        game_over = False
        # also if there is nothing happening for too long then the snake gets a reward
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        # place new food or just move -- 4
        if self.head == self.food:
            self.score += 1
            reward = 10
            # length of the snake increased because of no pop()
            self._place_food()
        else:
            # decide to randomly place a bomb
            if self.bomb and self.bomb_tick < self.bomb_time:
                self.bomb_tick += 1
            elif self.bomb and self.bomb_tick == self.bomb_time:
                self.bomb = None
                self.bomb_tick = 0
            elif not self.bomb:
                if random.random() > 0.9:
                    self._place_bomb()

            # move the snake by changing its length
            self.snake.pop()

        # update the UI and the clock -- 5
        self._update_ui()
        self.clock.tick(SPEED)

        # return the stat for the game
        return reward, game_over, self.score

    def _update_ui(self):
        # white background
        self.display.fill(WHITE)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLACK, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GRAY, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, GREEN, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        if self.bomb:
            pygame.draw.rect(self.display, RED, pygame.Rect(self.bomb.x, self.bomb.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, GRAY)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        # hits a bomb
        if pt == self.bomb:
            return True
        return False

    def _move(self, action):
        # get  direction via the action [straight, right, left] -> [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            # go right
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:
            # go left has to be [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)
