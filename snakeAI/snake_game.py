import pygame
import random
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.SysFont('arial', 25)


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
SPEED = 20  # ms


class SnakeGame:
    def __init__(self, w=720, h=680):
        self.w = w
        self.h = h

        # initial display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('snake')
        self.clock = pygame.time.Clock()
        self.bomb_time = 100

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

    def play_step(self):
        # get the user input -- 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            # check if any key is pressed
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                if event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                if event.key == pygame.K_UP:
                    self.direction = Direction.UP
                if event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN

        # move the snake -- 2
        self._move(self.direction)
        # changes the head
        self.snake.insert(0, self.head)

        # check if it is game over -- 3
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score

        # place new food or just move -- 4
        if self.head == self.food:
            self.score += 1
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
        return game_over, self.score

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

    def _is_collision(self):
        # hits boundary
        if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.h - BLOCK_SIZE or self.head.y < 0:
            return True
        # hits itself
        if self.head in self.snake[1:]:
            return True
        # hits a bomb
        if self.head == self.bomb:
            return True
        return False

    def _move(self, direction):
        x = self.head.x
        y = self.head.y

        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)


if __name__ == '__main__':
    game = SnakeGame()

    # game loop
    while True:
        game_over, score = game.play_step()
        if game_over:
            break

    print('Final Score', score)
    pygame.quit()
