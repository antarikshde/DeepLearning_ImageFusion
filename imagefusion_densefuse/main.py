# Demo - train the DenseFuse network & use it to generate an image

from __future__ import print_function

import time

from train_recons import train_recons
from generate import generate
from utils import list_images
import os

# True for training phase
IS_TRAINING = False
IS_TEST_RGB = True
BATCH_SIZE = 4
EPOCHES = 50

MODEL_SAVE_PATHS = [
    './models/densefuse_gray/densefuse_model_bs2_epoch4_all_weight_1e2.ckpt',
]

# In testing process, 'model_pre_path' is set to None
# The "model_pre_path" in "main.py" is just a pre-train model and not necessary for training and testing. 
# It is set as None when you want to train your own model. 
# If you already train a model, you can set it as your model for initialize weights.
model_pre_path = MODEL_SAVE_PATHS[0]

def main():

	if IS_TRAINING:

		inputPath = './images/train/'
		validatioinPath = './images/val/'

		print('\nBegin to train the network ...\n')
		train_recons(inputPath, validatioinPath, model_pre_path, model_pre_path, EPOCHES, BATCH_SIZE, debug=True)

		print('\nSuccessfully! Done training...\n')
	else:
		model_path = MODEL_SAVE_PATHS[0]
		print('\nBegin to generate pictures ...\n')
		path = 'images/test/color/1/'
		for i in range(1):
			index = i + 1
			path, dirs, files = next(os.walk(path))
			file_count = len(files)
			print('\n Number of Images: ', file_count)
			images = ["" for x in range(file_count)]
			i = 0
			for filename in files:
				images[i] = path + filename
				i += 1


			# choose fusion layer
			fusion_type = 'addition'
			output_save_path = 'outputs'
			generate(images, model_pre_path, model_pre_path,
					 index, IS_TEST_RGB, type = fusion_type, output_path = output_save_path)


if __name__ == '__main__':
    main()

