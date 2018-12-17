# Use a trained DenseFuse Net to generate fused images

import tensorflow as tf
import numpy as np
from datetime import datetime

from fusion_l1norm import L1_norm
from densefuse_net import DenseFuseNet
from utils import get_images, save_images, get_train_images, get_train_images_rgb


def generate(images_path, model_path, model_pre_path, index, isRGB, type='addition', output_path=None):
	if(isRGB):
		print('RGB - addition')
		_handler_rgb(images_path, model_path, model_pre_path, index, output_path=output_path)

		print('RGB - l1')
		_handler_rgb_l1(images_path, model_path, model_pre_path, index, output_path=output_path)
	else:
		if type == 'addition':
			print('addition')
			_handler(images_path, "", model_path, model_pre_path, index, output_path=output_path)
		elif type == 'l1':
			print('l1')
			_handler_l1(images_path, "", model_path, model_pre_path, index, output_path=output_path)


def _handler(ir_path, vis_path, model_path, model_pre_path, index, output_path=None):
	ir_img = get_train_images(ir_path, flag=False)
	vis_img = get_train_images(vis_path, flag=False)
	# ir_img = get_train_images_rgb(ir_path, flag=False)
	# vis_img = get_train_images_rgb(vis_path, flag=False)
	dimension = ir_img.shape

	ir_img = ir_img.reshape([1, dimension[0], dimension[1], dimension[2]])
	vis_img = vis_img.reshape([1, dimension[0], dimension[1], dimension[2]])

	ir_img = np.transpose(ir_img, (0, 2, 1, 3))
	vis_img = np.transpose(vis_img, (0, 2, 1, 3))

	print('img shape final:', ir_img.shape)

	with tf.Graph().as_default(), tf.Session() as sess:
		infrared_field = tf.placeholder(
			tf.float32, shape=ir_img.shape, name='content')
		visible_field = tf.placeholder(
			tf.float32, shape=ir_img.shape, name='style')

		dfn = DenseFuseNet(model_pre_path)

		output_image = dfn.transform_addition(infrared_field, visible_field)
		# restore the trained model and run the style transferring
		saver = tf.train.Saver()
		saver.restore(sess, model_path)

		output = sess.run(output_image, feed_dict={infrared_field: ir_img, visible_field: vis_img})

		save_images(ir_path, output, output_path, prefix='fused' + str(index), suffix='_densefuse_addition')


def _handler_l1(ir_path, vis_path, model_path, model_pre_path, index, output_path=None):
	ir_img = get_train_images(ir_path, flag=False)
	vis_img = get_train_images(vis_path, flag=False)
	dimension = ir_img.shape

	ir_img = ir_img.reshape([1, dimension[0], dimension[1], dimension[2]])
	vis_img = vis_img.reshape([1, dimension[0], dimension[1], dimension[2]])

	ir_img = np.transpose(ir_img, (0, 2, 1, 3))
	vis_img = np.transpose(vis_img, (0, 2, 1, 3))

	print('img shape final:', ir_img.shape)

	with tf.Graph().as_default(), tf.Session() as sess:

		# build the dataflow graph
		infrared_field = tf.placeholder(
			tf.float32, shape=ir_img.shape, name='content')
		visible_field = tf.placeholder(
			tf.float32, shape=ir_img.shape, name='style')

		dfn = DenseFuseNet(model_pre_path)

		enc_ir = dfn.transform_encoder(infrared_field)
		enc_vis = dfn.transform_encoder(visible_field)

		target = tf.placeholder(tf.float32, shape=enc_ir.shape, name='target')

		output_image = dfn.transform_decoder(target)

		# restore the trained model and run the style transferring
		saver = tf.train.Saver()
		saver.restore(sess, model_path)

		enc_ir_temp, enc_vis_temp = sess.run([enc_ir, enc_vis], feed_dict={infrared_field: ir_img, visible_field: vis_img})
		feature = L1_norm(enc_ir_temp, enc_vis_temp)

		output = sess.run(output_image, feed_dict={target: feature})
		save_images(ir_path, output, output_path, prefix='fused' + str(index), suffix='_densefuse_l1norm')

def _handler_rgb(images_path, model_path, model_pre_path, index, output_path=None):
	size = len(images_path)
	images = ["" for x in range(size)]
	ir_img1 = ["" for x in range(size)]
	ir_img2 = ["" for x in range(size)]
	ir_img3 = ["" for x in range(size)]
	for x in range(0, size):
		images[x] = get_train_images_rgb(images_path[x], flag=False)
		dimension = images[x].shape

		images[x] = images[x].reshape([1, dimension[0], dimension[1], dimension[2]])
		images[x] = np.transpose(images[x], (0, 2, 1, 3))

		ir_img1[x] = images[x][:, :, :, 0]
		ir_img1[x] = ir_img1[x].reshape([1, dimension[0], dimension[1], 1])
		ir_img2[x] = images[x][:, :, :, 1]
		ir_img2[x] = ir_img2[x].reshape([1, dimension[0], dimension[1], 1])
		ir_img3[x] = images[x][:, :, :, 2]
		ir_img3[x] = ir_img3[x].reshape([1, dimension[0], dimension[1], 1])
	print('img shape final:', ir_img1[0].shape)

	with tf.Graph().as_default(), tf.Session() as sess:
		images_field = ["" for x in range(size)]
		for x in range(0, size):
			images_field[x] = tf.placeholder(
				tf.float32, shape=ir_img1[0].shape)

		dfn = DenseFuseNet(model_pre_path)

		output_image = dfn.transform_addition(images_field)
		# restore the trained model and run the style transferring
		saver = tf.train.Saver()
		saver.restore(sess, model_path)

		output1 = sess.run(output_image, feed_dict={i: d for i, d in zip(images_field, ir_img1)})
		output2 = sess.run(output_image, feed_dict={i: d for i, d in zip(images_field, ir_img2)})
		output3 = sess.run(output_image, feed_dict={i: d for i, d in zip(images_field, ir_img3)})

		output1 = output1.reshape([1, dimension[0], dimension[1]])
		output2 = output2.reshape([1, dimension[0], dimension[1]])
		output3 = output3.reshape([1, dimension[0], dimension[1]])

		output = np.stack((output1, output2, output3), axis=-1)
		output = np.transpose(output, (0, 2, 1, 3))
		save_images(images_path, output, output_path, prefix='fused' + str(index), suffix='_densefuse_addition')


def _handler_rgb_l1(images_path, model_path, model_pre_path, index, output_path=None):
	size = len(images_path)
	images = ["" for x in range(size)]
	ir_img1 = ["" for x in range(size)]
	ir_img2 = ["" for x in range(size)]
	ir_img3 = ["" for x in range(size)]
	for x in range(0, size):
		images[x] = get_train_images_rgb(images_path[x], flag=False)
		dimension = images[x].shape

		images[x] = images[x].reshape([1, dimension[0], dimension[1], dimension[2]])

		images[x] = np.transpose(images[x], (0, 2, 1, 3))

		ir_img1[x] = images[x][:, :, :, 0]
		ir_img1[x] = ir_img1[x].reshape([1, dimension[0], dimension[1], 1])
		ir_img2[x] = images[x][:, :, :, 1]
		ir_img2[x] = ir_img2[x].reshape([1, dimension[0], dimension[1], 1])
		ir_img3[x] = images[x][:, :, :, 2]
		ir_img3[x] = ir_img3[x].reshape([1, dimension[0], dimension[1], 1])

	print('img shape final:', ir_img1[0].shape)

	with tf.Graph().as_default(), tf.Session() as sess:
		images_field = ["" for x in range(size)]
		for x in range(0, size):
			images_field[x] = tf.placeholder(
				tf.float32, shape=ir_img1[0].shape)

		dfn = DenseFuseNet(model_pre_path)
		enc_irs = ["" for x in range(size)]
		enc_irs = dfn.transform_encoder(images_field)

		target = tf.placeholder(
			tf.float32, shape=enc_irs[0].shape, name='target')

		output_image = dfn.transform_decoder(target)

		# restore the trained model and run the style transferring
		saver = tf.train.Saver()
		saver.restore(sess, model_path)

		enc_ir_temps = sess.run(enc_irs, feed_dict={i: d for i, d in zip(images_field, ir_img1)})
		feature = L1_norm(enc_ir_temps)
		output1 = sess.run(output_image, feed_dict={target: feature})

		enc_ir_temps = sess.run(enc_irs, feed_dict={i: d for i, d in zip(images_field, ir_img2)})
		feature = L1_norm(enc_ir_temps)
		output2 = sess.run(output_image, feed_dict={target: feature})

		enc_ir_temps = sess.run(enc_irs, feed_dict={i: d for i, d in zip(images_field, ir_img3)})
		feature = L1_norm(enc_ir_temps)
		output3 = sess.run(output_image, feed_dict={target: feature})

		output1 = output1.reshape([1, dimension[0], dimension[1]])
		output2 = output2.reshape([1, dimension[0], dimension[1]])
		output3 = output3.reshape([1, dimension[0], dimension[1]])

		output = np.stack((output1, output2, output3), axis=-1)
		output = np.transpose(output, (0, 2, 1, 3))
		save_images(images_path, output, output_path, prefix='fused' + str(index), suffix='_densefuse_l1norm')

def _handler_rgb_patch_based(images_path, model_path, model_pre_path, index, output_path=None):
	size = len(images_path)
	images = ["" for x in range(size)]
	ir_img1 = ["" for x in range(size)]
	ir_img2 = ["" for x in range(size)]
	ir_img3 = ["" for x in range(size)]
	for x in range(0, size):
		images[x] = get_train_images_rgb(images_path[x], flag=False)
		dimension = images[x].shape

		images[x] = images[x].reshape([1, dimension[0], dimension[1], dimension[2]])
		images[x] = np.transpose(images[x], (0, 2, 1, 3))

		ir_img1[x] = images[x][:, :, :, 0]
		ir_img1[x] = ir_img1[x].reshape([1, dimension[0], dimension[1], 1])
		ir_img2[x] = images[x][:, :, :, 1]
		ir_img2[x] = ir_img2[x].reshape([1, dimension[0], dimension[1], 1])
		ir_img3[x] = images[x][:, :, :, 2]
		ir_img3[x] = ir_img3[x].reshape([1, dimension[0], dimension[1], 1])
	print('img shape final:', ir_img1[0].shape)

	dimension = images[0].shape
	TRAIN_TAIL_SIZE_X = 32
	TRAIN_TAIL_SIZE_Y = 32
	TILES_X = (int)(dimension[1] / TRAIN_TAIL_SIZE_X)
	TILES_Y = (int)(dimension[2] / TRAIN_TAIL_SIZE_Y)
	INPUT_SHAPE_TILE = (1, TRAIN_TAIL_SIZE_X, TRAIN_TAIL_SIZE_Y, 1)


	with tf.Graph().as_default(), tf.Session() as sess:
		images_field = ["" for x in range(size)]
		for x in range(0, size):
			images_field[x] = tf.placeholder(
				tf.float32, shape=INPUT_SHAPE_TILE)

		dfn = DenseFuseNet(model_pre_path)

		output_image = dfn.transform_addition(images_field)
		# restore the trained model and run the style transferring
		saver = tf.train.Saver()
		saver.restore(sess, model_path)

		dimension = images[0].shape
		output1 = np.zeros([1, dimension[1], dimension[2]])
		output2 = np.zeros([1, dimension[1], dimension[2]])
		output3 = np.zeros([1, dimension[1], dimension[2]])
		for tile_x in range(TILES_X):
			for tile_y in range(TILES_Y):
				x1 = tile_x * TRAIN_TAIL_SIZE_X
				y1 = tile_y * TRAIN_TAIL_SIZE_Y
				x2 = x1 + TRAIN_TAIL_SIZE_X
				y2 = y1 + TRAIN_TAIL_SIZE_Y
				tile_img1 = ["" for x in range(size)]
				tile_img2 = ["" for x in range(size)]
				tile_img3 = ["" for x in range(size)]
				for x in range(0, size):
					tile_img1[x] = ir_img1[x][:, x1:x2, y1:y2, :]
					tile_img2[x] = ir_img2[x][:, x1:x2, y1:y2, :]
					tile_img3[x] = ir_img3[x][:, x1:x2, y1:y2, :]
				output1_t = sess.run(output_image, feed_dict={i: d for i, d in zip(images_field, tile_img1)})
				output2_t = sess.run(output_image, feed_dict={i: d for i, d in zip(images_field, tile_img2)})
				output3_t = sess.run(output_image, feed_dict={i: d for i, d in zip(images_field, tile_img3)})

				output1_t = output1_t.reshape([1, TRAIN_TAIL_SIZE_X, TRAIN_TAIL_SIZE_Y])
				output2_t = output2_t.reshape([1, TRAIN_TAIL_SIZE_X, TRAIN_TAIL_SIZE_Y])
				output3_t = output3_t.reshape([1, TRAIN_TAIL_SIZE_X, TRAIN_TAIL_SIZE_Y])
				output1[0, x1:x2, y1:y2] = output1_t
				output2[0, x1:x2, y1:y2] = output2_t
				output3[0, x1:x2, y1:y2] = output3_t

		output = np.stack((output1, output2, output3), axis=-1)
		output = np.transpose(output, (0, 2, 1, 3))
		save_images(images_path, output, output_path, prefix='fused' + str(index), suffix='_densefuse_addition_patch_based')
