import json

import matplotlib.pyplot as plt
import numpy as np

from bottleneck import Bottleneck
from cnn import build_cnn
from identification import get_scores, calc_metrics
from tpe import build_tpe

n_in = 256
n_out = 256

with open('data/meta.json', 'r') as f:
    meta = json.load(f)

cnn = build_cnn(227, meta['n_subjects'])
cnn.load_weights('data/weights/weights.best.h5')
bottleneck = Bottleneck(cnn, ~1)

train_x, train_y = np.load('data/train_x.npy'), np.load('data/train_y.npy')
dev_x = np.load('data/dev_x.npy')
dev_protocol = np.load('data/dev_protocol.npy')

train_emb = bottleneck.predict(train_x, batch_size=256)
dev_emb = bottleneck.predict(dev_x, batch_size=256)

del train_x

W_pca = np.load('data/w_pca.npy')

tpe, tpe_pred = build_tpe(n_in, n_out, W_pca.T)

train_y = np.array(train_y)
subjects = list(set(train_y))

tpe.load_weights('data/weights/weights.tpe.mineer.h5')

dev_emb2 = tpe_pred.predict(dev_emb)

protocol = np.load('data/dev_protocol.npy')
tsc, isc = get_scores(dev_emb2, protocol)
eer, fars, frrs, dists = calc_metrics(tsc, isc)

print('EER: {}'.format(eer * 100))

plt.figure()
plt.hist(tsc, 20, color='g', normed=True, alpha=0.3)
plt.hist(isc, 20, color='r', normed=True, alpha=0.3)

plt.figure()
plt.loglog(fars, frrs)
plt.show()

for a, b, c in zip(fars, frrs, dists):
    print('a: {:.2f} | r: {:.2f} | d: {:.2f}'.format(a, b, c))
