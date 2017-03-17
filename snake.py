import time
import numpy as np
import termin

dirs = 'up down left right'.split()
up, down, left, right = dirs
up_i, down_i, left_i, right_i = range(len(dirs))
dir_vectors = [np.r_[ 0, -1],
               np.r_[ 0, +1],
               np.r_[-1,  0],
               np.r_[+1,  0]]
board_items = 'empty snake_segment apple wall'.split()
empty, snake_segment, apple, wall = board_items
empty_i, snake_segment_i, apple_i, wall_i = range(len(board_items))

class GameOver(Exception): pass
class Loss(GameOver): pass
class Win(GameOver): pass

class game(object):
    apple_extension = 1
    segment_score = 5
    apples = 1

    def __init__(game, board, position, extensions=2):
        game.board = board.astype(np.uint8)
        game.position = np.r_[position]
        game.segments = [game.position]
        game.extensions = extensions
        game.current_cell = snake_segment
        for i in range(game.apples):
            game.put_apple()

    @classmethod
    def from_size(cls, width, height, **kw):
        return cls(np.zeros((width, height)), (width//2, height//2), **kw)

    def is_in_bounds(game, x, y):
        height, width = game.board.shape
        return x >= 0 and y >= 0 and x < width and y < height

    def cell(game, x, y):
        return board_items[game.board[y, x]]

    def set_cell(game, x, y, item):
        game.board[y, x] = board_items.index(item)

    @property
    def current_cell(game):
        x, y = game.position
        return game.cell(x, y)

    @current_cell.setter
    def current_cell(game, item):
        x, y = game.position
        game.set_cell(x, y, item)

    @property
    def is_over(game):
        return not np.allclose(game.segments[-1], game.position) \
            or not game.is_in_bounds(*game.position)

    @property
    def score(game):
        return len(game.segments)*game.segment_score

    def next(game, action):
        game.position = game.position + dir_vectors[dirs.index(action)]
        if not game.is_in_bounds(*game.position):
            raise Loss('outside bounds')
        elif game.current_cell == snake_segment:
            raise Loss('ate self')
        elif game.current_cell == apple:
            game.move_snake()
            game.eat_apple()
        else:
            game.move_snake()

    def eat_apple(game):
        game.extensions += game.apple_extension
        height, width = game.board.shape
        if np.any(game.board == empty_i):
            game.put_apple()

    def random_cell(game, item=empty):
        x, y = np.nonzero(game.board == board_items.index(item))
        i = np.random.randint(x.shape[0])
        return np.r_[y[i], x[i]]

    def put_apple(game):
        x, y = game.random_cell()
        game.set_cell(x, y, apple)

    def move_snake(game):
        game.current_cell = snake_segment
        game.segments.append(game.position)
        if game.extensions:
            game.extensions -= 1
        else:
            x, y = game.segments.pop(0)
            game.set_cell(x, y, empty)

    def __str__(game):
        t = ''.join('|{}|\n'.format(''.join(' X@'[c] for c in r)) for r in game.board)
        divider = '+' + game.board.shape[1]*'=' + '+\n'
        return divider + t + divider

def main(width=5, height=5):
    g = game.from_size(width, height)
    with termin.inputs() as inputs:
        direction = right
        while not g.is_over:
            print(g.board)
            time.sleep(0.5)
            inp = direction
            while inp is not None:
                inp = next(inputs)
                direction = inp if inp in dirs else direction
            try:
                g.next(direction)
            except GameOver as e:
                print('game over!', *e.args)
                print('score:', g.score)

if __name__ == "__main__":
    main()
