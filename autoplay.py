import sys
import time
import numpy as np
import snake

def play(mod, *args):
    phi, g = None, snake.game.from_size(mod.W, mod.H)
    while not g.is_over:
        print(g)
        time.sleep(0.1)
        phi = mod.next_phi(phi, g)
        q = mod.q(phi, *args)
        a = sample_softmax(q)
        try:
            g.next(snake.dirs[a])
        except snake.GameOver as e:
            print('game over!', *e.args)
            print('q for chosen action:', q[a])
            print('q for   UDLR action:', q)
            print('score:', g.score)

def sample_softmax(v, temperature=2e-1):
    p  = np.exp(v/temperature)
    p /= p.sum()
    return np.random.choice(len(snake.dirs), p=p)

if __name__ == "__main__":
    mod = __import__(sys.argv[1])
    args = mod.init(sys.argv[2:])
    play(mod, *args)
