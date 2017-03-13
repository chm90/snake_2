import sys
import time
import numpy as np

import snake
import qlearn

W, H = 6, 6  

def play(Q):
    g = snake.game.from_size(W, H)
    while not g.is_over:
        print(g.board)
        time.sleep(0.1)
        a = ply(Q, g)
        try:
            g.next(a)
        except snake.GameOver as e:
            print('game over!', *e.args)
            print('score:', g.score)

def ply(Q, g, temperature=1e-1):
    s = qlearn.state(g)
    q_s = Q[(qlearn.wholeaxis,) + tuple(s)]
    p_s = np.exp(q_s/temperature)
    p_s /= p_s.sum()
    a = np.random.choice(len(snake.dirs), p=p_s)
    return snake.dirs[a]

if __name__ == "__main__":
    Q = np.load(sys.argv[1])
    print('num empty Q:', 100.*np.count_nonzero(Q == 0.0)/Q.size, '%')
    play(np.load(sys.argv[1]))
