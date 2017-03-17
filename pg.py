import sys
from functools import reduce
from keras import layers, models, regularizers
from keras.layers import convolutional
import numpy as np
import cnake

# T for timesteps, H for height, W for width, L for layers.
T, H, W, L = 3, 64, 64, 4
input_shape = T,H,W,L
phi_states = T
action_dim = 2

print('Constructing models', end='... ', flush=True, file=sys.stderr)

in_state = layers.Input(shape=input_shape)

# Trendy stanza is convolve, normalize, non-linearize, pool, dropout. We
# use 48 kernels of size 3x3x4 on a 128x128x4x3 input, giving us 128x128x1x48.
# We drop the third dimension. This is then normalized, relu'd, and max pooled
# by a 2x2 giving us 62x62x48.
h0 = layers.MaxPooling2D(pool_size=(2, 2), border_mode='same')(
     layers.Activation('elu')(
     layers.BatchNormalization()(
     layers.Reshape((W-2, H-2, 48))(
     convolutional.Convolution3D(48, T, 3, 3, border_mode='valid')(in_state)))))

# More of the same, we have bigly inputs! YUGE! Dropout is nice I hear at 25%.
# Output size is now 30x30x48.
h1 = layers.Dropout(0.25)(
     layers.MaxPooling2D(pool_size=(2, 2), border_mode='same')(
     layers.Activation('elu')(
     layers.BatchNormalization()(
     convolutional.Convolution2D(48, 3, 3, border_mode='valid')(h0)))))

# Our 30x30x48 input is now convolved with 64 kernels of size 3x3, giving us
# 30x30x64. Batch normalize, relu. Do it again with 64 more kernels. Max
# pool with 2x2 stride, finally 15x15x64. Dropout at 25%.
h2 = layers.Dropout(0.25)(
     layers.MaxPooling2D(pool_size=(2, 2), border_mode='same')(
     layers.Activation('elu')(
     layers.BatchNormalization()(
     convolutional.Convolution2D(64, 3, 3, border_mode='same')(
     layers.Activation('elu')(
     layers.BatchNormalization()(
     convolutional.Convolution2D(64, 3, 3, border_mode='same')(h1))))))))

# Our state is now 15x15x64, or 14 400 dimensions. Hot damn.
state_out = layers.Flatten()(h2)

actor_action_layer = \
    layers.Dense(action_dim, name='actor_a')(
    layers.Dense(128, activation='relu', name='actor_h1')(
    layers.BatchNormalization(name='actor_bn0')(
    layers.Dense(32, activation='relu', name='actor_h0')(state_out))))

q_layers = [
    layers.Dense(128, activation='relu', name='critic_h0'),
    layers.BatchNormalization(name='critic_bn0'),
    layers.Dense(32, activation='relu', name='critic_h1'),
    layers.Dense(1, name='critic_q')]

def wrap_q_layers(in_action):
    in_merge = layers.merge([state_out, in_action], mode='concat')
    return reduce(lambda result, layer: layer(result), q_layers, in_merge)

in_action = layers.Input((action_dim,))

print('done', file=sys.stderr)

print('Compiling models', end='... ', flush=True, file=sys.stderr)

# Plain actor model
actor = models.Model(input=in_state, output=actor_action_layer)
actor.compile(optimizer='adam', loss='mean_squared_error')

# Plain critic model
critic = models.Model(input=[in_state, in_action], output=wrap_q_layers(in_action))
critic.compile(optimizer='adam', loss='mean_squared_error')

# actor_q is a special model where the loss function is simply -y_pred,
# independent of the ground truth to be fitted against. This means that we
# optimize the predicted value itself, maximizing it. The predicted value here
# is the Q value for the action the actor network would take. In other words,
# it will optimize the actor network so that the action's Q value is maximized.
#
# What's more, we don't want to optimize the critic just because when
# optimizing the actor network we make use of the critic. Therefore we "freeze"
# the layers belonging to the critic.

actor_q = models.Model(input=in_state, output=wrap_q_layers(actor_action_layer))

for layer in actor_q.layers:
    if hasattr(layer, 'trainable') and layer not in critic.layers:
        layer.trainable = False

actor_q.compile(optimizer='adam', loss=lambda y, yp: -yp)

print('done', file=sys.stderr)

def init(args):
    h5_fn, = args
    actor_h5_fn, critic_h5_fn = args
    actor.load_weights(actor_h5_fn)
    critic.load_weights(critic_q_h5_fn)
    return (actor, critic)

def state(g):
    return np.random.rand(H, W, L)

def phi(phi, g):
    s = state(g)
    if phi is None:
        return [s]*phi_states
    else:
        phi.append(s)
        return phi[-phi_states:]

def action(phi, actor, critic):
    return actor.predict(np.array([phi]))[0]

def copy_model(model):
    return models.model_from_yaml(model.to_yaml())

def main(args=sys.argv[1:], alpha=0.0, action_sigma=1e-2, num_batch=50,
         num_copy_target=2000, num_iter=int(2e6), num_replay=int(1e6),
         replay_period=4, gamma=0.5, t_last_reward_max=25):

    actor_h5_fn, critic_h5_fn = args + ['actor.h5', 'critic.h5'][:2 - len(args)]

    print('Creating target networks', end='... ', flush=True, file=sys.stderr)
    actor_target = copy_model(actor)
    critic_target = copy_model(critic)
    print('done', file=sys.stderr)

    print('Loading weights', end='... ', flush=True, file=sys.stderr)
    try:
        actor.load_weights(actor_h5_fn)
        critic.load_weights(critic_q_h5_fn)
    except IOError as e:
        print(e, file=sys.stderr)
    else:
        print('done', file=sys.stderr)

    actor_target.set_weights(actor.get_weights())
    critic_target.set_weights(critic.get_weights())

    # A replay tuple is (a, r, terminal, phi, phi_).
    replay_a = np.zeros((num_replay, action_dim))
    replay_r = np.zeros(num_replay)
    replay_p = np.ones(num_replay)
    replay_terminal = np.zeros(num_replay, dtype=np.bool)
    replay_phi  = np.zeros((num_replay,) + input_shape)
    replay_phi_ = np.zeros((num_replay,) + input_shape)

    g, replay_i, Rtot, high_score = None, 0, 0, 0

    for i in range(num_iter):

        if g is None or g.is_over:
            g = cnake.game.from_size(W, H)
            t, t_last_reward = 0, 0
            phi = [state(g)]*phi_states
            phi_ = list(phi)

        phi.append(state(g))
        phi = phi[-phi_states:]
        score = g.score

        # Take deterministic action with some exploration noise
        a  = actor.predict(np.array([phi]))[0]
        a += np.random.normal(0, action_sigma, size=action_dim)

        # Cast the dice!
        try:
            g.next(a)
        except cnake.GameOver as e:
            pass

        if g.score > high_score:
            high_score = g.score
            print('New high score:', high_score)

        phi_.append(state(g))
        phi_ = phi_[-phi_states:]
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
        if i < num_batch or (i % replay_period) != 0:
            continue

        print('Begin minibatch', end='... ', flush=True, file=sys.stderr)

        # Sample a minibatch
        jmax = min(i, num_replay)
        mb_idxs = np.random.choice(jmax, p=replay_p[:jmax]/replay_p[:jmax].sum(),
                                   size=num_batch, replace=False)
        mb_phi  = replay_phi[mb_idxs]
        mb_phi_ = replay_phi_[mb_idxs]
        mb_a    = replay_a[mb_idxs]
        mb_nonterm, = np.nonzero(replay_terminal[mb_idxs] == 0)

        # Predict Q of phi and phi_ for minibatch
        mb_q  = critic.predict([mb_phi, mb_a])[:, 0]
        mb_a_ = actor_target.predict(mb_phi_)
        mb_q_ = critic_target.predict([mb_phi_, mb_a_])[:, 0]

        # Calculate Q's to fit against for critic network from target critic
        # network using the Bellman equation:
        #   y_j = r_j + ɣ \max_a' Q(s', a'),
        # where the maximum is approxmated using the target critic network.
        mb_y              = replay_r[mb_idxs].copy()
        mb_y[mb_nonterm] += gamma*mb_q_[mb_nonterm]

        # Update importance weight using TD error ∂, p = (ε + |∂|)^α.
        replay_p[mb_idxs] = (1.0 + np.abs(mb_y - mb_q))**alpha

        # Finally learn. First optimize critic, then optimize actor_q.
        L  = critic.train_on_batch([mb_phi, mb_a], mb_y)
        L += actor_q.train_on_batch(mb_phi, np.zeros(mb_y.shape))

        # Soft update target networks.
        #print(critic_target.get_weights())
        #critic_target.soft_update(critic)
        #actor_target.soft_update(actor)

        print('done', file=sys.stderr)

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
