import keras.backend as K
import numpy as np
from keras.layers import Dense, Lambda, Input, subtract, multiply
from keras.models import Model, Sequential


def triplet_loss(_, y_pred):
    return -K.mean(K.log(K.sigmoid(y_pred)))


def build_tpe(n_in, n_out, weights=None):
    a = Input(shape=(n_in,))
    p = Input(shape=(n_in,))
    n = Input(shape=(n_in,))

    if weights is None:
        weights = np.zeros((n_in, n_out))

    base_model = Sequential()
    base_model.add(Dense(n_out, input_dim=n_in, use_bias=False, weights=[weights], activation='linear'))
    base_model.add(Lambda(lambda x: K.l2_normalize(x, axis=1)))

    a_emb = base_model(a)
    p_emb = base_model(p)
    n_emb = base_model(n)

    e = K.sum(multiply([a_emb, subtract([p_emb, n_emb])]), axis=1)

    model = Model([a, p, n], e)
    predict = Model(a, a_emb)

    model.compile(loss=triplet_loss, optimizer='rmsprop')

    return model, predict
