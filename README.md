# Mask-detection-RCNN
### Side: Trained a mask detection model using Faster RCNN (Pytorch).
> This project is originally done on kaggle notebook and now transfered to github. [Link](https://www.kaggle.com/jackttai/face-mask-detection-faster-r-cnn-pytorch)

## Dataset
- The image data is downloaded from kaggle: https://www.kaggle.com/andrewmvd/face-mask-detection
- Remember to modifiy the dataset path in `configs.yaml` after downloading the files.

## Major dependencies
- pillow
- opencv
- torch
- torchvision
- numpy
- pyyaml
- pandas

## Main files:
- `main.py`
- `tools.py`
- `train.py`
- `test.py`
- `dataset.py`
- `configs.yaml`

## Steps:
1. Create the environment, install the main dependencies
2. Activate the environment, run python `main.py`, if some libaries are missed, install it with `pip`.
3. Modify the config in `configs.yaml`, such as learning rate, epochs and folder path.
4. Start training and wait for finishing.
5. Use `test.py` to do single image prediction and see the result.


