#!/bin/bash
python ../train/training.py\
	--dataset=driven\
	--image_path=../../../../../datadrive/train/images\
	--mask_path=../../../../../datadrive/train/masks\
	--lr=3e-3\
	--wd=1e-6\
	--epochs=50\
	--shuffle=True\
	--batch_size=4\
	--save_model=True
