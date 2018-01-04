import numpy as np
import snake
#
# Provide a openai gym-like interface for our snake environment
#

class Discrete(object):
    def __init__(self, n):
        self.n = n
        self.rng = np.random.RandomState()
    def sample(self):
        return self.rng.randint(self.n)
    def contains(self, x):
        try:
            x = int(x)
        except:
            return False
        return x >= 0 and x < self.n

    def shape(self):
        return (self.n,)

# Env-related abstractions
class Env(object):
    def __init__(self, shape, action_space, seed=None):
        self.seed_ = seed
        self.shape = shape
        self.action_space = action_space
        self.seed(seed)

    def step(self, action):
        score = self.game.score
        info = {}
        try:
            self.game.next(snake.dirs[action])
        except snake.GameOver as e:
            info['gameover!'] = str(*e.args)

        new_score = self.game.score
        reward = float(-10 if self.game.is_over else new_score - score)

        # (observation, reward, terminal, info) in accordance with gym api
        return (self.game.state(), reward, self.game.is_over, info)

    def reset(self):
        start_pos = (self.shape[0]//2, self.shape[1]//2)
        self.game = snake.game(np.zeros(self.shape), start_pos, seed=self.seed_)

        return (self.game.board, 0, False, None)

    def render(self, mode='human', close=False):
        raise NotImplementedError

    def close(self):
        self.reset()

    def seed(self, seed=None):
        if seed:
            self.seed_ = seed
        return self.seed_


def make(env='Snake-v0', shape=(5,5), num_actions=4, seed=None):
    if not seed:
        seed = np.random.randint(0, 2**31)
    return Env(shape, Discrete(num_actions), seed=seed)

if __name__ == '__main__':
    import time
    # random snake test
    env = make(shape=(10,10))
    obs, reward, done, info = env.reset()
    while not done:
        print(env.game)
        a = env.action_space.sample()
        print("snake moves", snake.dirs[a])
        _, reward, done, info = env.step(a)
        time.sleep(0.5)

    for k,v in info.items():
        print(k,v)


