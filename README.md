# Face identification using CNN + TPE

**NOTE: This repository is archived and will no longer be updated.**

This repository contains an implementation of the 
[Triplet Probabilistic Embedding for Face Verification and Clustering](https://arxiv.org/abs/1604.05417) paper.

![demo application screenshot](https://storage.yandexcloud.net/meownoid-pro-static/external/github/face-identification-tpe/screenshot.png)

### Installation

```shell script
git clone https://github.com/meownoid/face-identification-tpe.git
cd face-identification-tpe
python -m pip install -r requirements.txt
```

### Usage

**NOTE: Pre-trained model was trained using very small dataset and achieves poor performance. It can't be used in
any real-world application and is intended for education purposes only.**

To start application with the pre-trained weights download all
[assets](https://yadi.sk/d/zIWpWyX73ACTAg) and put them to the `model` directory (default path) or
to the any other directory.
 
Then you can start the application.
 
 ```shell script
python application.py
```

If you placed assets to the other directory, specify path with the `--model-path` argument.

 ```shell script
python application.py --model-path /path/to/assets/
```

### Training

**NOTE: Training code was written a long time ago and have a lot of hard-coded constants in it.
Using it now on new dataset will be very difficult, so please, don't try. You can read it and use it as a reference
or you can just use CNN and TPE definitions and write custom training code.**

**I'm leaving this here just for the sake of history.**

1. Download assets `face_template.npy` and `shape_predictor_68_face_landmarks.dat` 
from [here](https://yadi.sk/d/zIWpWyX73ACTAg) and put them to the `model` dir.

2. Place train, test and evaluation (named `dev`) data to the `data` folder using following structure.
```
data\
    dev\
        person_0\
            1.jpg
            2.jpg
            ...
        person_1\
            1.jpg
            2.jpg
            ...
        ...
    test\
        person_0\
            1.jpg
            2.jpg
            ...
        person_1\
            1.jpg
            2.jpg
            ...
        ...
    train\
        person_0\
            1.jpg
            2.jpg
            ...
        person_1\
            1.jpg
            2.jpg
            ...
        ...
```

All images in the `person_{i}` folder inside `train` and `test` directories
must contain faces of the same person.

3. Run `python 0_load_data.py`
4. Train the CNN with `python 1_train_cnn.py`
5. Optionally test the CNN with `python 2_test_cnn.py`
6. Train the TPE with `python 3_train_tpe.py`
7. Optionally test the TPE with `python 4_test_tpe.py`

