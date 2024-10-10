# CNN_DEMO

## Running predict.py

1. Install tensorflow with conda: https://www.tensorflow.org/install/pip
1. After installing tensorflow and activating the `tf` environment, install
```bash
pip install pillow
pip install mtcnn
pip install opencv-python
pip install numpy==1.26.4
```
1. Clone this repo
1. Download the h5 model and save it in the same directory as the rest of the files in this repo
1. In the conda terminal, with the tf environment activated, run
```bash
cd /path/to/CNN
python predict.py
```

## Running training.py

1. Download the data from https://www.kaggle.com/datasets/sudarshanvaidya/random-images-for-face-emotion-recognition
1. Clone this repo if not already done so
1. Make following folder structure
```
CNN
  - data
    - 1-train
      - anger
      - happy
      - sad
      - surprised
      - ....
    - 2-test
      - anger
      - happy
      - sad
      - surprised
      - ....
    - 3-validate
      - anger
      - happy
      - sad
      - surprised
      - ....
```
1. Place all the original images in the train folder, in their sub categories.
1. Move 30 images of each category to the folders in test, and then move another 30 images of each category to validate
1. Install tensorflow with conda if not already installed: https://www.tensorflow.org/install/pip
1. In the conda terminal, with the tf environment activated, run `python train.py`
1. Models will be saved at the end of training
