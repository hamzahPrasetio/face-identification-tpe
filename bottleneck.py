import numpy as np
from keras.models import Model


class Bottleneck:
    def __init__(self, model, layer):
        self.model = Model(model.inputs, model.layers[layer].output)

    def predict(self, data_x, batch_size=32):
        n_data = len(data_x)
        n_batches = n_data // batch_size + (0 if n_data % batch_size == 0 else 1)

        result = []

        for i in range(n_batches):
            batch_x = np.vstack([x[np.newaxis] for x in data_x[i * batch_size:(i + 1) * batch_size]])
            result.append(self.model([batch_x, 0]).numpy())

        return np.vstack(result)
