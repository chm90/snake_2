import sys
from keras import layers, models, regularizers
import numpy as np
import snake, autoplay

H, W = 5, 5

# W*H snake segment positions, then the apple's position.
input_dim = W*H + 1

# Keep in mind these are mirrored in the decoder side.
layer_sizes = 15, 5

encoded_dim = layer_sizes[-1]

layer_input   = layers.Input(shape=(input_dim,))

layer_encoded = layer_input
for size in layer_sizes:
    layer_encoded = layers.Dense(size, activation='relu',
                                 activity_regularizer=regularizers.activity_l2(1e-4)
                                )(layer_encoded)

layer_decoded = layer_encoded
for size in layer_sizes:
    layer_decoded = layers.Dense(size,
                                 activation='relu',
                                 activity_regularizer=regularizers.activity_l2(1e-4)
                                )(layer_decoded)
layer_decoded = layers.Dense(input_dim, activation='sigmoid')(layer_decoded)

# Whole autoencoder pipeline; this is what we train.
model_autoencoder = models.Model(input=layer_input, output=layer_decoded)

# Only the encoder, useful for obtaining the encoded representations.
model_encoder = models.Model(input=layer_input, output=layer_encoded)

# Only the decoder, useful for getting the state from encoded representations.
layer_input_encoded = layers.Input(shape=(encoded_dim,))
layer_decoded_output = model_autoencoder.layers[-1](layer_input_encoded)
model_decoder = models.Model(input=layer_input_encoded,
                             output=layer_decoded_output)

def state(g):
    w, h = g.board.shape
    s = np.zeros(w*h + 1)
    segs = (1.0 + np.array(g.segments).dot((1, w))) / float(w*h)
    s[:segs.size] = segs
    if np.any(g.board == snake.apple_i):
        s[-1] = (1.0 + g.random_cell(item=snake.apple).dot((1, w))) / float(w*h)
    return s

def main(args=sys.argv[1:]):
    q_mat_fn, = args
    Q = np.load(q_mat_fn)

    print('Compiling model', end='... ', file=sys.stderr)
    model_autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    print('done', file=sys.stderr)

    num_train, num_test = 80000, 20000
    num_total = num_train + num_test
    print('Sampling', num_total, 'states', end='..', file=sys.stderr)
    x = np.zeros((num_total, input_dim))
    g = None
    for i in range(num_total):
        if g is None:
            g = snake.game.from_size(W, H)
        x[i, :] = state(g)
        try:
            g.next(autoplay.ply(Q, g))
        except snake.GameOver:
            g = None
        if (i % (num_total//10)) == 0:
            print(end='.', file=sys.stderr)
    print(' done', file=sys.stderr)

    np.random.shuffle(x)
    model_autoencoder.fit(x, x,
                          nb_epoch=500,
                          batch_size=1024,
                          shuffle=True,
                          validation_split=num_test/num_total)

    model_autoencoder.save('autoencoder.h5')

if __name__ == "__main__":
    main()
