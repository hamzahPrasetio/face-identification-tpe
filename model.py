import os.path

import numpy as np

from bottleneck import Bottleneck
from cnn import build_cnn
from preprocessing import FaceDetector, FaceAligner, clip_to_range
from tpe import build_tpe

GREATER_THAN = 32
BATCH_SIZE = 128
IMAGE_SIZE = 217
IMAGE_BORDER = 5


class FaceVerificatorError(Exception):
    pass


class FaceVerificator:
    def __init__(self, model_path):
        self._model_path = model_path

        self._model_files = {
            'shape_predictor': os.path.join(model_path, 'shape_predictor_68_face_landmarks.dat'),
            'face_template': os.path.join(model_path, 'face_template.npy'),
            'mean': os.path.join(model_path, 'mean.npy'),
            'stddev': os.path.join(model_path, 'stddev.npy'),
            'cnn_weights': os.path.join(model_path, 'weights_cnn.h5'),
            'tpe_weights': os.path.join(model_path, 'weights_tpe.h5'),
        }

        for model_file in self._model_files.values():
            if not os.path.exists(model_file):
                raise FileNotFoundError(model_file)

        self._mean = np.load(self._model_files['mean'])
        self._stddev = np.load(self._model_files['stddev'])
        self._fd = FaceDetector()
        self._fa = FaceAligner(
            self._model_files['shape_predictor'],
            self._model_files['face_template']
        )
        cnn = build_cnn(227, 266)
        cnn.load_weights(self._model_files['cnn_weights'])
        self._cnn = Bottleneck(cnn, ~1)
        _, tpe = build_tpe(256, 256)
        tpe.load_weights(self._model_files['tpe_weights'])
        self._tpe = tpe

    def normalize(self, img):
        img = clip_to_range(img)
        return (img - self._mean) / self._stddev

    def process_image(self, img):
        face_rects = self._fd.detect_faces(img, upscale_factor=2, greater_than=GREATER_THAN)

        if not face_rects:
            return []

        faces = self._fa.align_faces(img, face_rects, dim=IMAGE_SIZE, border=IMAGE_BORDER)
        faces = list(map(self.normalize, faces))

        faces_y = self._cnn.predict(faces, batch_size=BATCH_SIZE)
        faces_y = self._tpe.predict(faces_y, batch_size=BATCH_SIZE)

        return list(zip(face_rects, faces_y))

    @staticmethod
    def compare_many(dist, xs, ys):
        xs = np.array(xs)
        ys = np.array(ys)
        scores = xs @ ys.T
        return scores, scores > dist
