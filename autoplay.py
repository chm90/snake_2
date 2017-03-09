import sys
import time
import numpy as np

import snake
import rl

W, H = 6, 6  

def play(Q):
    g = snake.game(np.zeros((W, H)), (W//2, H//2))
    while not g.is_over:
        print(g.board)
        time.sleep(0.1)
        a = ply(Q, g)
        try:
            g.next(a)
        except snake.GameOver as e:
            print('game over!', *e.args)
            print('score:', g.score)

def ply(Q, g, epsilon=1.0-2e-3):
    s = rl.state(g)
    q_s = Q[(rl.wholeaxis,) + tuple(s)]
    if np.random.random() < epsilon and np.any(q_s != 0):
        a = q_s.argmax()
    else:
        a = np.random.choice(len(snake.dirs))
    return snake.dirs[a]

if __name__ == "__main__":
    play(np.load(sys.argv[1]))
