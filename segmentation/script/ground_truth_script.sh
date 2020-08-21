#!/bin/bash

python ../train/training.py --dataset=groundtruth --filepath=../../../segmentation_ground_truth --lr=1e-3 --wd=1e-5 --epochs=3 --shuffle=True --batch_size=4
