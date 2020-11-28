import json
import os.path

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import OneHotEncoder

from cnn import build_cnn

WEIGHTS_DIR = 'data/weights/'
NB_EPOCH = 100
BATCH_SIZE = 32
AUGMENTATION = True

if not os.path.exists(WEIGHTS_DIR):
    os.mkdir(WEIGHTS_DIR)

oh = OneHotEncoder()

train_x, train_y = np.load('data/train_x.npy'), np.load('data/train_y.npy')
test_x, test_y = np.load('data/test_x.npy'), np.load('data/test_y.npy')

n_subjects = len(set(train_y))
n_train = train_x.shape[0]
n_test = test_x.shape[0]

oh.fit(train_y.reshape(-1, 1))

train_y = oh.transform(train_y.reshape(-1, 1)).todense()
test_y = oh.transform(test_y.reshape(-1, 1)).todense()

print('n_train: {}'.format(n_train))
print('n_test: {}'.format(n_test))
print('n_subjects: {}'.format(n_subjects))

with open('data/meta.json', 'w') as f:
    json.dump({'n_subjects': n_subjects}, f)

mc1 = ModelCheckpoint(
    WEIGHTS_DIR + 'weights.best.h5',
    monitor='val_accuracy',
    verbose=0,
    save_best_only=True,
    mode='max'
)

model = build_cnn(227, n_subjects)
model.summary()

weights_to_load = WEIGHTS_DIR + 'weights.best.h5'

if os.path.exists(weights_to_load):
    model.load_weights(weights_to_load)

try:
    if AUGMENTATION:
        data_gen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.1,
            horizontal_flip=True
        )

        model.fit_generator(
            data_gen.flow(train_x, train_y, batch_size=BATCH_SIZE),
            steps_per_epoch=train_x.shape[0],
            epochs=NB_EPOCH,
            validation_data=(test_x, test_y),
            callbacks=[mc1]
        )
    else:
        model.fit(
            train_x, train_y,
            batch_size=BATCH_SIZE,
            epochs=NB_EPOCH,
            validation_data=(test_x, test_y),
            callbacks=[mc1]
        )
finally:
    model.save_weights(WEIGHTS_DIR + 'weights.finally.h5')
