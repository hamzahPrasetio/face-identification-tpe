import os
import os.path
from itertools import repeat

import numpy as np
from skimage import io
from tqdm import tqdm

from preprocessing import FaceDetector, FaceAligner, clip_to_range

fd = FaceDetector()
fa = FaceAligner('model/shape_predictor_68_face_landmarks.dat', 'model/face_template.npy')

MODEL_DIR = 'model'
DATA_DIR = 'data'
IMAGE_FORMATS = {'.jpg', '.jpeg', '.png'}


def get_images(path):
    return list(filter(
        lambda s: os.path.isfile(os.path.join(path, s)) and os.path.splitext(s)[1] in IMAGE_FORMATS,
        os.listdir(path)
    ))


def get_folders(path):
    return list(filter(lambda s: os.path.isdir(os.path.join(path, s)), os.listdir(path)))


def list_data(data_path):
    result = []
    persons = get_folders(data_path)
    persons.sort()

    for person in persons:
        person_dir = os.path.join(data_path, person)
        person_files = get_images(person_dir)
        person_files.sort()
        person_files = list(map(lambda x: os.path.join(person_dir, x), person_files))
        person_id = int(person.split('_')[1])
        result.extend(zip(person_files, repeat(person_id)))

    return result


def load_file(filename, image_size=96, border=0):
    total_size = image_size + 2 * border

    img = io.imread(filename)
    faces = fd.detect_faces(img, get_top=1)

    if len(faces) == 0:
        return None

    face = fa.align_face(img, faces[0], dim=image_size, border=border).reshape(1, total_size, total_size, 3)
    face = clip_to_range(face)

    return face.astype(np.float32)


def load_data(
        data,
        not_found_policy='throw_away',
        available_subjects=None,
        image_size=96,
        border=0
):
    n_data = len(data)

    total_size = image_size + 2 * border

    images = np.zeros((n_data, total_size, total_size, 3), dtype=np.float32)
    labels = np.zeros((n_data,), dtype=np.int)

    if available_subjects is not None:
        available_subjects = set(available_subjects)

    black = np.zeros((1, total_size, total_size, 3), dtype=np.float32)

    face_not_found_on = []

    img_ptr = 0
    for filename, subject in tqdm(data):
        if available_subjects is not None:
            if subject not in available_subjects:
                continue

        face_img = load_file(filename, image_size=image_size, border=border)

        if face_img is None:
            face_not_found_on.append(filename)
            if not_found_policy == 'throw_away':
                continue
            elif not_found_policy == 'replace_black':
                face_img = black
            else:
                raise Exception('Face not found on {}'.format(filename))

        images[img_ptr] = face_img
        labels[img_ptr] = subject
        img_ptr += 1

    images = images[:img_ptr]
    labels = labels[:img_ptr]

    if len(face_not_found_on) > 0:
        print('[Warning] Faces was not found on:')
        for f in face_not_found_on:
            print(' - {}'.format(f))

    return images, labels


IMAGE_SIZE = 217
BORDER = 5


def main():
    print('Loading train files...')
    train_files = list_data(os.path.join(DATA_DIR, 'train'))
    train_x, train_y = load_data(
        train_files,
        image_size=IMAGE_SIZE,
        border=BORDER,
        not_found_policy='throw_away'
    )

    mean = train_x.mean(axis=0)
    stddev = train_x.std(axis=0)

    np.save(os.path.join(MODEL_DIR, 'mean'), mean)
    np.save(os.path.join(MODEL_DIR, 'stddev'), stddev)

    train_x -= mean
    train_x /= stddev

    np.save(os.path.join(DATA_DIR, 'train_x'), train_x)
    np.save(os.path.join(DATA_DIR, 'train_y'), train_y)

    del train_x

    print('Loading test files...')
    test_files = list_data(os.path.join(DATA_DIR, 'test'))
    test_x, test_y = load_data(
        test_files,
        image_size=IMAGE_SIZE,
        border=BORDER,
        not_found_policy='throw_away',
        available_subjects=train_y
    )

    del train_y

    test_x -= mean
    test_x /= stddev

    np.save(os.path.join(DATA_DIR, 'test_x'), test_x)
    np.save(os.path.join(DATA_DIR, 'test_y'), test_y)

    del test_x, test_y

    print('Loading dev files...')
    dev_files = list_data(os.path.join(DATA_DIR, 'dev'))
    dev_x, dev_y = load_data(
        dev_files,
        image_size=IMAGE_SIZE,
        border=BORDER,
        not_found_policy='replace_black'
    )

    dev_x -= mean
    dev_x /= stddev

    np.save(os.path.join(DATA_DIR, 'dev_x'), dev_x)

    dev_protocol = (np.repeat(dev_y[:, np.newaxis], len(dev_y), axis=1) == dev_y)

    np.save(os.path.join(DATA_DIR, 'dev_protocol'), dev_protocol)


if __name__ == '__main__':
    main()
