import os
import sys
from keras import layers, models, regularizers, initializers, optimizers
from keras.layers import convolutional
import numpy as np
import snake

# T for timesteps, H for height, W for width, L for layers.
T, H, W, L = 3, 32, 32, 3
input_shape = H,W,T*L
phi_states = T
action_dim = 4
action_repeat = T
epsilon = 15e-2

print('Constructing & compiling models', end='... ', flush=True, file=sys.stderr)

in_state = layers.Input(shape=input_shape)

# Trendy stanza is convolve, normalize, non-linearize, pool, dropout. We
# use 48 kernels of size 3x3x4 on a 128x128x4x3 input, giving us 128x128x1x48.
# We drop the third dimension. This is then normalized, relu'd, and max pooled
# by a 2x2 giving us 62x62x48.
c0 = layers.Activation('relu')(
     layers.BatchNormalization()(
     convolutional.Conv2D(16, (8, 8), strides=(4, 4), padding='valid')(in_state)))

# More of the same, we have bigly inputs! YUGE! Dropout is nice I hear at 25%.
# Output size is now 30x30x48.
c1 = layers.Activation('relu')(
     layers.BatchNormalization()(
     convolutional.Conv2D(32, (4, 4), strides=(2, 2), padding='valid')(c0)))

h_layers = \
    layers.Dense(action_dim,
                 kernel_regularizer=regularizers.l2(),
                 activation='linear',
                 name='q')(
    layers.BatchNormalization(name='bn0')(
    layers.Dense(256,
                 kernel_regularizer=regularizers.l2(),
                 activation='relu',
                 name='h0')(
    layers.Flatten()(c1))))

model = models.Model(inputs=in_state, outputs=h_layers)
model.compile(optimizer=optimizers.Adam(lr=1e-5), loss='mean_squared_error')

print('done', file=sys.stderr)

def state(g):
    s = np.zeros((W, H, L))
    for i in range(L):
        s[:, :, i] = 1.0*(g.board == i)
    return s

def init(args):
    h5_fn, = args
    model.load_weights(h5_fn)
    return (model,)

def next_phi(phi, g):
    s = state(g)
    if phi is None:
        phi = np.repeat(s, T, axis=-1)
    else:
        phi[:, :, :-L] = phi[:, :, L:]
        phi[:, :, -L:] = s
    return phi

def q(phi, model):
    return model.predict(np.array([phi]))[0, :]

def soft_update(dst, src, tau=1e-3):
    dst.set_weights([tau*w_src + (1.0 - tau)*w_dst
                    for w_src, w_dst in zip(src.get_weights(), dst.get_weights())])

def main(args=sys.argv[1:], alpha=0.7, epsilon=5e-2, num_batch=50,
         num_copy_target=2000, num_iter=int(2e6), num_replay=int(1e5),
         replay_period=4, gamma=0.5, t_last_reward_max=25):

    h5_fn, = args

    print('Creating target network', end='... ', flush=True, file=sys.stderr)
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

    # A replay tuple is (a, r, terminal, phi, phi_) with importance weight p.
    replay_a = -np.ones(num_replay, dtype=np.int8)
    replay_r = np.zeros(num_replay)
    replay_p = np.ones(num_replay)
    replay_terminal = np.zeros(num_replay, dtype=np.bool)
    replay_phi  = np.zeros((num_replay,) + input_shape)
    replay_phi_ = np.zeros((num_replay,) + input_shape)

    g, replay_i, Rtot, high_score, num_games = None, 0, 0, 0, 0

    for t in range(num_iter):

        if g is None or g.is_over:
            g = snake.game.from_size(W, H, apples=int(np.sqrt(W*H)))
            num_games += 1
            t_start, t_last_reward = t, t
            phi_ = next_phi(None, g)
            phi = phi_.copy()

        phi[:] = phi_
        score = g.score

        # ε-greedy policy
        if np.random.random() < epsilon:
            a = np.random.choice(len(snake.dirs))
        else:
            a = q(phi, model).argmax()

        # Cast the dice!
        try:
            g.next(snake.dirs[a])
        except snake.GameOver as e:
            pass

        if g.score > high_score:
            high_score = g.score
            print('New high score:', high_score)

        phi_ = next_phi(phi_, g)
        r = float(-10 if g.is_over else g.score - score)

        # Track time of last reward. Abandon cowardly policies.
        t += 1
        t_last_reward = t if r > 0 else t_last_reward
        if t - t_last_reward >= t_last_reward_max:
            g = None

        Rtot += r

        # Record in replay memory
        replay_i = (replay_i + 1) % num_replay
        if replay_i == 0:
            print('Replay buffer full, reset loop index')
        replay_a[replay_i] = a
        replay_r[replay_i] = r
        replay_p[replay_i] = replay_p.max()
        replay_terminal[replay_i] = g is None or g.is_over
        replay_phi[replay_i] = phi
        replay_phi_[replay_i] = phi_

        # Don't do learning until we have at least some experience, and only
        # each replay_period'th iteration.
        if t < num_batch or (t % replay_period) != 0:
            continue

        # Sample a minibatch
        jmax = min(t, num_replay)
        mb_idxs = np.random.choice(jmax, p=replay_p[:jmax]/replay_p[:jmax].sum(),
                                   size=num_batch, replace=False)
        mb_phi  = replay_phi[mb_idxs, :, :]
        mb_phi_ = replay_phi_[mb_idxs, :, :]
        mb_a    = replay_a[mb_idxs]
        mb_nonterm, = np.nonzero(replay_terminal[mb_idxs] == 0)

        # Predict Q of s and s_ for minibatch
        mb_q  = model.predict(mb_phi)
        mb_q_ = target.predict(mb_phi_)

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
        L = model.train_on_batch(replay_phi[mb_idxs, :, :], mb_y)

        # Soft update target networks.
        soft_update(dst=target, src=model)

        if (t % num_copy_target) == 0:
            print((' -- Saving DQN (#{t}) --\n'
                   '        Average R: {rpg:.3g} in {num_games} games\n'
                   '       High score: {high_score}\n'
                   '    Training loss: {L}\n'
                   ' Minibatch mean Q: {mean_Q}\n'
                   ' Minibatch mean y: {mean_y}\n').format(
                  rpg=Rtot/num_games,
                  mean_Q= np.r_[mb_q].mean(axis=0),
                  mean_y=mb_y.mean(axis=0), **locals()))
            model.save(h5_fn)
            Rtot, num_games = 0, 0

    model.save(h5_fn)

if __name__ == "__main__":
    main()
