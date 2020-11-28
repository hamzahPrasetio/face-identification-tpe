from keras.layers import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import PReLU

from keras.models import Sequential


def build_cnn(dim, n_classes):
    model = Sequential()

    model.add(Conv2D(
        96, 11, 4,
        input_shape=(dim, dim, 3),
        kernel_initializer='glorot_uniform',
        padding='same'
    ))
    model.add(PReLU())
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))

    model.add(Conv2D(
        256, 5, 1,
        kernel_initializer='glorot_uniform',
        padding='same'
    ))
    model.add(PReLU())
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))

    model.add(Conv2D(
        384, 3, 1,
        kernel_initializer='glorot_uniform',
        padding='same'
    ))
    model.add(PReLU())

    model.add(Conv2D(
        384, 3, 1,
        kernel_initializer='glorot_uniform',
        padding='same'
    ))
    model.add(PReLU())

    model.add(Conv2D(
        256, 3, 1,
        kernel_initializer='glorot_uniform',
        padding='same'
    ))
    model.add(PReLU())
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(2048, kernel_initializer='glorot_uniform'))
    model.add(PReLU())
    model.add(Dropout(0.5))

    model.add(Dense(256, kernel_initializer='glorot_uniform'))
    model.add(PReLU())

    model.add(Dense(n_classes, kernel_initializer='glorot_uniform', activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
