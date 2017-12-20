import sys
import numpy as np
import snake

# SNAKE Q STATE SPACE
#
# Snake is high-dimensional. Sad! To work around this, look at the neighborhood
# of the snake's head. With looking at one step around the snake, we will have
# nine cells that can have three states, so 4^9 states with four actions gives
# us 4^10 = 8 MB. The states are empty, snake_segment and apple. We always add
# the apple.

W, H = 5, 5
N, M = len(snake.dirs), len(snake.board_items)
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

#def state(g):
#    s = [g.board[py, px] for px, py in np.r_[1, 1] + idxs]
#    if np.all(np.abs(g.position - 1) <= 1):
#        pos_idx = idxs.index(tuple(g.position - 1))
#        s[pos_idx] = snake.wall_i
#    return tuple(s)

#board = np.array(
#[[0,1,1,0,0],
# [0,1,1,0,0],
# [1,1,1,0,0],
# [1,1,0,0,0],
# [0,0,0,0,2]]
#)
#g = snake.game(board, (1, 3))
#s = state(g)
#print(np.r_[s].reshape((3,3)))
#raise SystemExit

def explore(Q, high_score=None):
    g = snake.game(np.zeros((W, H)), (W//2, H//2))
    state_actions = []
    while not g.is_over:
        s = state(g)
        q_s = Q[(wholeaxis,) + tuple(s)]
        if np.random.random() < epsilon and np.any(q_s != 0):
            a = q_s.argmax()
        else:
            a = np.random.choice(N)
        score = g.score
        try:
            g.next(snake.dirs[a])
        except snake.GameOver as e:
            pass
        s_ = state(g)
        r = float(-10 if g.is_over else g.score - score)
        yield r, (a,) + s, (wholeaxis,) + s_

def learn_Q(t_max=1000, output='Q', save_interval=1000):
    Q = np.zeros((N,) + (M,)*len(idxs))
    Rtot, Rmax = 0, 0
    for i in range(200000):
        #episode = list(explore(Q))
        #R = sum(r for r, sa, s_a in episode)
        #for t, (r, sa, s_a) in enumerate(episode):
        #    Q[sa] = (1.0 - learning_rate)*Q[sa] + learning_rate*((gamma**t)*R)
        R = 0
        for t, (r, sa, s_a) in enumerate(explore(Q)):
            if t >= t_max:
                break
            best_next_Q = Q[s_a].max()
            Q_new = r + gamma*best_next_Q
            Q[sa] = (1.0 - learning_rate)*Q[sa] + learning_rate*Q_new
            R += r
        Rtot += R
        if R > Rmax:
            Rmax = R
            print('New high score:', Rmax)
            #print(np.r_[s].reshape((3,3)))
        if (i % save_interval) == 0:
            print(r'Saving Q, \avg R =', Rtot/1000.)
            np.save(output, Q)
            Rtot = 0
    return Q

def q(s, Q):
    return Q[(wholeaxis,) + tuple(s)]

def load(args):
    Q = np.load(sys.argv[1])
    print('num empty Q:', 100.*np.count_nonzero(Q == 0.0)/Q.size, '%')
    return (Q,)

def phi(phi, g):
    return state(g)

if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else 'Q'
    Q = learn_Q(output=arg)
