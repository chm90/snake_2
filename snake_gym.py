import numpy as np
import snake
# Env-related abstractions

class Env(object):
    def __init__(self, width, height, seed=None):
        if not seed:
            seed = np.random.randint(0, 2**31)
        self.seed = seed
        self.shape = (width, height)

    def step(self, action):
        g = self.game
        score = g.score
        try:
            self.game.next(snake.dirs[action])
        except:
            pass

        r = float(-10 if g.is_over else g.score - score)

        return (g.state(), r, g.is_over, None)

    def reset(self):
        start_pos = (self.shape[0]//2, self.shape[1]//2)
        self.game = snake.game(np.zeros(self.shape), start_pos, seed=self.seed)

        return (self.game.state(), 0, False, None)

    def render(self, mode='human', close=False):
        raise NotImplementedError

    def close(self):
        reset()

    def seed(self, seed=None):
        if seed:
            self.seed = seed
        return self.seed


def make(env='Snake-v0',W=5,H=5,seed=None):
    if not seed:
        seed = np.random.randint(0, 2**31)
    return Env(W,H,seed)

if __name__ == '__main__':
    env = make()
    ob, r, d, _ = env.reset()
    ob, r, d, _ = env.step(0)

    print(str(ob))

