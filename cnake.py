"continuous action snake"
import numpy as np
import snake
from snake import GameOver

class game(snake.game):
    apples = 10

    def next(g, a):
        a = np.argmax([dv.dot(a)/a.dot(a)/dv.dot(dv)
                       for dv in snake.dir_vectors])
        return super(game, g).next(snake.dirs[a])
