import os
import sys
from keras import layers, models, regularizers, optimizers
import numpy as np
import snake

H, W = 5, 5

# W*H snake segment positions, then the apple's position.
input_dim = W*H + 1

# Number of historical states to use for Q network input.
theta_states = 3

# Dense layer sizes
layer_sizes = 1024, 512, len(snake.dirs)

layer_input = layers.Input(shape=(theta_states, input_dim))
layer = layers.Flatten()(layer_input)
for size in layer_sizes:
    layer = layers.Dense(size, activation='relu',
                         activity_regularizer=regularizers.activity_l2(1e-6)
                         )(layer)

model = models.Model(input=layer_input, output=layer)

loss = 'mean_squared_error'
optimizer = optimizers.Adadelta(lr=0.8)

def state(g):
    w, h = g.board.shape
    s = np.zeros(w*h + 1)
    segs = (1.0 + np.array(g.segments).dot((1, w))) #/ float(w*h)
    s[:segs.size] = segs
    if np.any(g.board == snake.apple_i):
        s[-1] = (1.0 + g.random_cell(item=snake.apple).dot((1, w))) #/ float(w*h)
    return s

def main(args=sys.argv[1:], alpha=0.7, epsilon=5e-2, num_batch=500,
         num_copy_target=2000, num_iter=int(2e6), num_replay=int(1e6),
         replay_period=4, gamma=0.8):
    h5_fn, = args
    print('Compiling model', end='... ', file=sys.stderr)
    model.compile(optimizer=optimizer, loss=loss)
    target = models.model_from_yaml(model.to_yaml())
    print('done', file=sys.stderr)

    print('Loading weights', end='... ', file=sys.stderr)
    try:
        model.load_weights(h5_fn)
    except IOError as e:
        print(e, file=sys.stderr)
    else:
        print('done', file=sys.stderr)

    target.set_weights(model.get_weights())

    # A replay tuple is (a, r, terminal, theta, theta_).
    replay_a = -np.ones(num_replay, dtype=np.int8)
    replay_r = np.zeros(num_replay)
    replay_p = np.ones(num_replay)
    replay_terminal = np.zeros(num_replay, dtype=np.bool)
    replay_theta  = np.zeros((num_replay, theta_states, input_dim))
    replay_theta_ = np.zeros((num_replay, theta_states, input_dim))

    g, replay_i, Rtot, high_score = None, 0, 0, 0

    for i in range(num_iter):

        if g is None or g.is_over:
            g = snake.game.from_size(W, H)
            theta = [state(g)]*theta_states

        theta.append(state(g))
        theta = theta[-theta_states:]
        score = g.score

        # ε-greedy policy
        if np.random.random() < epsilon:
            a = np.random.choice(len(snake.dirs))
        else:
            q = model.predict(np.array([theta]))
            a = q.argmax()

        # Cast the dice!
        try:
            g.next(snake.dirs[a])
        except snake.GameOver:
            pass

        if g.score > high_score:
            high_score = g.score
            print('New high score:', high_score)

        theta_ = theta + [state(g)]
        theta_ = theta_[-theta_states:]
        r = float(-10 if g.is_over else g.score - score)
        Rtot += r

        # Record in replay memory
        replay_i = (replay_i + 1) % num_replay
        if replay_i == 0:
            print('Replay buffer full, reset loop index')
        replay_a[replay_i] = a
        replay_r[replay_i] = r
        replay_p[replay_i] = replay_p.max()
        replay_terminal[replay_i] = g.is_over
        replay_theta[replay_i, :, :] = theta
        replay_theta_[replay_i, :, :] = theta_

        # Don't do learning until we have at least some experience, and only
        # each replay_period'th iteration.
        if i < num_batch or (i % replay_period) != 0:
            continue

        # Sample a minibatch
        jmax = min(i, num_replay)
        mb_idxs = np.random.choice(jmax, p=replay_p[:jmax]/replay_p[:jmax].sum(),
                                   size=num_batch, replace=False)
        mb_theta  = replay_theta[mb_idxs, :, :]
        mb_theta_ = replay_theta_[mb_idxs, :, :]
        mb_a      = replay_a[mb_idxs]
        mb_nonterm, = np.nonzero(replay_terminal[mb_idxs] == 0)

        # Predict Q of s and s_ for minibatch
        mb_q  = model.predict(mb_theta)
        mb_q_ = target.predict(mb_theta_)

        # Set targets for minibatch, setting the target for each
        # action-state to the predicted value, except for the chosen action
        # where we set it to what amounts to be the counterpart of the
        # Bellman equation: y_{j,a} = r_j + \gamma max_{a_} Q(s_, a_).
        mb_y = mb_q.copy()
        js = np.arange(mb_y.shape[0])
        mb_y[js,             mb_a]              = replay_r[mb_idxs]
        mb_y[js[mb_nonterm], mb_a[mb_nonterm]] += gamma*(mb_q_[mb_nonterm, :].max(axis=1))

        # Update TD error
        replay_p[mb_idxs] = (1.0 + np.abs(mb_y[js, mb_a] - mb_q[js, mb_a]))**alpha

        # Finally learn
        L = model.train_on_batch(replay_theta[mb_idxs, :, :], mb_y)

        if (i % num_copy_target) == 0:
            target.set_weights(model.get_weights())
            print(' -- Saving DQN --')
            print('        Average R:', Rtot/num_copy_target)
            print('       High score:', high_score)
            print('    Training loss:', L)
            print(' Minibatch mean Q:', np.r_[mb_q].mean(axis=0))
            print(' Minibatch mean y:', mb_y.mean(axis=0))
            model.save(h5_fn)
            Rtot = 0

    model.save(h5_fn)

if __name__ == "__main__":
    main()
