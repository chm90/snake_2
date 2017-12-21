
import numpy as np
import snake
import sys

N, M = len(snake.dirs), len(snake.board_items)
W, H = 5, 5
epsilon, learning_rate, gamma = 0.995, 0.99, 0.5
wholeaxis = slice(None)
idxs = [(dx, dy) for dy in (-1, 0, +1) for dx in (-1, 0, +1)]

def state(g):
    x, y = g.position
    s = np.r_[[g.board[py, px] if g.is_in_bounds(px, py) else snake.wall_i
               for px, py in g.position + idxs]]
    if np.any(g.board == snake.apple_i):
        dp = g.random_cell(item=snake.apple) - g.position
        dp = np.fmin(+1, np.fmax(-1, dp))
        apple_idx = idxs.index(tuple(dp))
        if s[apple_idx] == snake.empty_i:
            s[apple_idx] = snake.apple_i
    return tuple(s)


def replay(seed, actions):
    g = snake.game(np.zeros((W,H)), (W//2,H//2), seed=seed)
    for a in actions:
        try:
            g.next(snake.dirs[a])
            print(g)
        except snake.GameOver as e:
            break

    print("score: " + str(g.score) + " actions: " + str(len(actions)))


def test_qlearn(npy,n=10):
    Q = np.load(npy)
    best = None
    for i in range(0, n):
        seed = np.random.randint(0, 2**31)
        g = snake.game(np.zeros((W,H)), (2,2), seed=seed)
        actions = []
        t = 0
        while not g.is_over and t < 10000:
            s = state(g)
            q_s = Q[(wholeaxis,) + tuple(s)]
            score = g.score
            if np.random.random() < epsilon and np.any(q_s != 0):
                a = q_s.argmax()
            else:
                a = np.random.choice(N)

            actions.append(a)

            try:
                g.next(snake.dirs[a])
            except snake.GameOver as e:
                pass

            r = float(-10 if g.is_over else g.score - score)
            t += 1

        if best is None or g.score > best[0].score:
            best = (g, actions)

    return best

def test_qlearn_gym(npy, n=10):
    import snake_gym as gym
    Q = np.load(npy)
    best = None
    for i in range(0, n):
        env = gym.make('Snake-v0')
        s, _, done, _ = env.reset()
        g = env.game
        actions = []
        t = 0
        while not done and t < 10000:
            q_s = Q[(wholeaxis,) + tuple(s)]
            score = g.score
            if np.random.random() < epsilon and np.any(q_s != 0):
                a = q_s.argmax()
            else:
                a = np.random.choice(N)

            s, r, done, _ = env.step(a)
            actions.append(a)
            t += 1

        if best is None or g.score > best[0].score:
            best = (g, actions)

    return best

if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else 'Q.npy'
    game, actions = test_qlearn_gym(arg, 1)
    replay(game.seed, actions)

