# Face identification using CNN + TPE

**NOTE: This repository is archived and will no longer be updated.**

This repository contains an implementation of the 
[Triplet Probabilistic Embedding for Face Verification and Clustering](https://arxiv.org/abs/1604.05417) paper.

![demo app screenshot](https://habrastorage.org/files/f83/9d3/057/f839d305744d45e69660baf2c0986ce6.png)

### Installation

```shell script
git clone https://github.com/meownoid/face-identification-tpe.git
cd face-identification-tpe
python -m pip install -r requirements.txt
```

### Usage

**NOTE:** Pre-trained model was trained using very small dataset and achieves poor performance. It can't be used in
any real-world application and is intended for education purposes only.

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
Download the `face_template.npy` and `shape_predictor_68_face_landmarks.dat` from [here](https://yadi.sk/d/zIWpWyX73ACTAg) and put them to the `model` dir.

Place training data in following order:
```
data\
    dev_protocol.npy
    dev\
        1.jpg
        2.jpg
        3.jpg
        ...
    test\
        subject_0\
            1.jpg
            2.jpg
            ...
        subject_1\
            1.jpg
            2.jpg
            ...
        ...
    train\
        subject_0\
            1.jpg
            2.jpg
            ...
        subject_1\
            1.jpg
            2.jpg
            ...
        ...
```
Then run as follows:

1. utils/load_data.py
2. train_cnn.py
3. train_tpe.py

Use the test scripts to test your model.
