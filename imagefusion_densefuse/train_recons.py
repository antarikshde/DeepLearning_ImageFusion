# Train the DenseFuse Net

from __future__ import print_function

import scipy.io as scio
import numpy as np
import math
import tensorflow as tf
import matplotlib.pyplot as plt

from ssim_loss_function import SSIM_LOSS
from densefuse_net import DenseFuseNet
from utils import get_train_images
from utils import list_images
from utils import list_folders
STYLE_LAYERS  = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1')

TRAINING_IMAGE_SHAPE = (256, 256, 1) # (height, width, color_channels)
TRAINING_IMAGE_SHAPE_OR = (256, 256, 1) # (height, width, color_channels)

LEARNING_RATE = 1e-4
EPSILON = 1e-5


def train_recons(inputPath, validationPath, save_path, model_pre_path, EPOCHES_set, BATCH_SIZE, debug=False, logging_period=1):
    from datetime import datetime
    start_time = datetime.now()
    path = './models/performanceData/'
    fileName = 'TrainPerformanceData_'+str(start_time)+'.txt'
    fileName = fileName.replace(" ", "_")
    fileName = fileName.replace(":", "_")
    file = open(path+fileName, 'w')
    file.close()
    folders = list_folders(inputPath)
    valFolders = list_folders(validationPath)
    EPOCHS = EPOCHES_set
    print("EPOCHES   : ", EPOCHS)
    print("BATCH_SIZE: ", BATCH_SIZE)
    # get the traing image shape
    HEIGHT, WIDTH, CHANNELS = TRAINING_IMAGE_SHAPE
    INPUT_SHAPE = (BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)

    HEIGHT_OR, WIDTH_OR, CHANNELS_OR = TRAINING_IMAGE_SHAPE_OR
    INPUT_SHAPE_OR = (BATCH_SIZE, HEIGHT_OR, WIDTH_OR, CHANNELS_OR)
    GROUNDTRUTH_SHAPE_OR = (1, HEIGHT_OR, WIDTH_OR, CHANNELS_OR)

    # create the graph
    with tf.Graph().as_default(), tf.Session() as sess:
        original = tf.placeholder(tf.float32, shape=INPUT_SHAPE_OR, name='original')
        groundtruth = tf.placeholder(tf.float32, shape=GROUNDTRUTH_SHAPE_OR, name='groundtruth')
        source = original

        print('source  :', source.shape)
        print('original:', original.shape)
        print('groundtruth:', groundtruth.shape)
        # create the deepfuse net (encoder and decoder)
        dfn = DenseFuseNet(model_pre_path)
        generated_img = dfn.transform_recons_train(source)
        print('generate:', generated_img.shape)
        pixel_loss = tf.reduce_sum(tf.square(groundtruth - generated_img))
        pixel_loss = tf.math.sqrt(pixel_loss / (BATCH_SIZE * HEIGHT * WIDTH))
        loss = pixel_loss
        train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        sess.run(tf.global_variables_initializer())

        # saver = tf.train.Saver()
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

        # ** Start Training **
        step = 0
        count_loss = 0
        numTrainSite = len(folders)
        numValSite = len(valFolders)

        for epoch in range(EPOCHS):
            save_path_epoc = './models_intermediate/'+str(epoch)+'.ckpt'
            start_time_epoc = datetime.now()
            for site in range(numTrainSite):
                start_time_site = datetime.now()
                file = open(path + fileName, 'a')
                groundtruth_imgs_path = list_images(inputPath+folders[site] + '/gt/')
                training_imgs_path = list_images(inputPath+folders[site])
                np.random.shuffle(training_imgs_path)
                gt = get_train_images(groundtruth_imgs_path, crop_height=HEIGHT, crop_width=WIDTH, flag=False)
                gtImgTrain = np.zeros(GROUNDTRUTH_SHAPE_OR)
                gtImgTrain[0] = gt
                n_batches = int(len(training_imgs_path) // BATCH_SIZE)
                for batch in range(n_batches):
                    original_path = training_imgs_path[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE)]
                    original_batch = get_train_images(original_path, crop_height=HEIGHT, crop_width=WIDTH, flag=False)
                    original_batch = original_batch.reshape([BATCH_SIZE, 256, 256, 1])
                    sess.run(train_op, feed_dict={original: original_batch, groundtruth: gtImgTrain})
                if debug:
                    for batch in range(n_batches):
                        original_path = training_imgs_path[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE)]
                        original_batch = get_train_images(original_path, crop_height=HEIGHT, crop_width=WIDTH, flag=False)
                        original_batch = original_batch.reshape([BATCH_SIZE, 256, 256, 1])

                        # print('original_batch shape final:', original_batch.shape)

                        # run the training step
                        _p_loss = sess.run(pixel_loss, feed_dict={original: original_batch, groundtruth: gtImgTrain})
                        # add text file to add  mode(validation/training), epoch#, site#, batch#, _p_loss) ------------------------------
                        file.write('Train[Epoch#: %d, Site#: %d, Batch#: %d, _p_loss: %d]\n' % (epoch, site, batch, _p_loss))
                        print('Train[Epoch#: %d, Site#: %d, Batch#: %d, _p_loss: %d]' % (epoch, site, batch, _p_loss))
                print('Time taken per site: %s' %(datetime.now() - start_time_site))
                file.close()
            for site in range(numValSite):
                file = open(path + fileName, 'a')
                start_time_validation = datetime.now()
                groundtruth_val_imgs_path = list_images(validationPath+valFolders[site] + '/gt/')
                validation_imgs_path = list_images(validationPath+valFolders[site])
                np.random.shuffle(validation_imgs_path)
                gt = get_train_images(groundtruth_val_imgs_path, crop_height=HEIGHT, crop_width=WIDTH, flag=False)
                gtImgVal = np.zeros(GROUNDTRUTH_SHAPE_OR)
                gtImgVal[0] = gt
                val_batches = int(len(validation_imgs_path) // BATCH_SIZE)
                val_pixel_acc = 0
                for batch in range(val_batches):
                    val_original_path = validation_imgs_path[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE)]
                    val_original_batch = get_train_images(val_original_path, crop_height=HEIGHT, crop_width=WIDTH, flag=False)
                    val_original_batch = val_original_batch.reshape([BATCH_SIZE, 256, 256, 1])

                    val_pixel = sess.run(pixel_loss, feed_dict={original: val_original_batch, groundtruth: gtImgVal})
                    file.write('Validation[Epoch#: %d, Site#: %d, Batch#: %d, _p_loss: %d]\n' % (epoch, site, batch, val_pixel))
                    val_pixel_acc = val_pixel_acc + val_pixel
                print('Time taken per validation site: %s' % (datetime.now() - start_time_validation))
                val_loss = val_pixel_acc/val_batches
                file.write('ValidationAcc[Epoch#: %d, Site#: %d, Batch#: %d, val_loss: %d]\n' % (epoch, site, batch, val_loss))
                print('ValidationAcc[Epoch#: %d, Site#: %d, Batch#: %d, _p_loss: %d]' % (epoch, site, batch, val_loss))
                file.close()
            print('------------------------------------------------------------------------------')
            print('Time taken per epoc: %s' % (datetime.now() - start_time_epoc))
            saver.save(sess, save_path_epoc)
        saver.save(sess, save_path)
        print('Done training!')
        print('Total Time taken (training): %s' % (datetime.now() - start_time))
        file.close()
    # ** Done Training & Save the model **

def train_recons_patch_based(inputPath, validationPath, save_path, model_pre_path, EPOCHES_set, BATCH_SIZE, debug=False, logging_period=1):
    from datetime import datetime
    start_time = datetime.now()
    path = './models/performanceData/'
    fileName = 'TrainPerformanceData_'+str(start_time)+'.txt'
    fileName = fileName.replace(" ", "_")
    fileName = fileName.replace(":", "_")
    file = open(path+fileName, 'w')
    file.close()
    folders = list_folders(inputPath)
    valFolders = list_folders(validationPath)
    EPOCHS = EPOCHES_set
    print("EPOCHES   : ", EPOCHS)
    print("BATCH_SIZE: ", BATCH_SIZE)
    # get the traing image shape
    HEIGHT, WIDTH, CHANNELS = TRAINING_IMAGE_SHAPE
    INPUT_SHAPE = (BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)

    HEIGHT_OR, WIDTH_OR, CHANNELS_OR = TRAINING_IMAGE_SHAPE_OR
    INPUT_SHAPE_OR = (BATCH_SIZE, HEIGHT_OR, WIDTH_OR, CHANNELS_OR)
    GROUNDTRUTH_SHAPE_OR = (1, HEIGHT_OR, WIDTH_OR, CHANNELS_OR)
    TRAIN_TAIL_SIZE_X = 32
    TRAIN_TAIL_SIZE_Y = 32

    TILES_X = (int)(TRAINING_IMAGE_SHAPE_OR[0]/TRAIN_TAIL_SIZE_X)
    TILES_Y = (int)(TRAINING_IMAGE_SHAPE_OR[0] / TRAIN_TAIL_SIZE_Y)
    INPUT_SHAPE_TILE = (BATCH_SIZE, TRAIN_TAIL_SIZE_X, TRAIN_TAIL_SIZE_Y, CHANNELS_OR)
    GT_INPUT_SHAPE_TILE = (1, TRAIN_TAIL_SIZE_X, TRAIN_TAIL_SIZE_Y, CHANNELS_OR)

    # create the graph
    with tf.Graph().as_default(), tf.Session() as sess:
        original = tf.placeholder(tf.float32, shape=INPUT_SHAPE_TILE, name='original')
        groundtruth = tf.placeholder(tf.float32, shape=GT_INPUT_SHAPE_TILE, name='groundtruth')
        source = original

        print('source  :', source.shape)
        print('original:', original.shape)
        print('groundtruth:', groundtruth.shape)
        # create the deepfuse net (encoder and decoder)
        dfn = DenseFuseNet(model_pre_path)
        generated_img = dfn.transform_recons_train(source)
        print('generate:', generated_img.shape)
        pixel_loss = tf.reduce_sum(tf.square(groundtruth - generated_img))
        pixel_loss = pixel_loss / (BATCH_SIZE * HEIGHT * WIDTH)
        loss = pixel_loss
        train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        sess.run(tf.global_variables_initializer())

        # saver = tf.train.Saver()
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

        # ** Start Training **
        step = 0
        count_loss = 0
        numTrainSite = len(folders)
        numValSite = len(valFolders)



        for epoch in range(EPOCHS):
            save_path_epoc = './models_intermediate/'+str(epoch)+'.ckpt'
            start_time_epoc = datetime.now()
            for site in range(numTrainSite):
                start_time_site = datetime.now()
                file = open(path + fileName, 'a')
                groundtruth_imgs_path = list_images(inputPath+folders[site] + '/gt/')
                training_imgs_path = list_images(inputPath+folders[site])
                np.random.shuffle(training_imgs_path)
                gt = get_train_images(groundtruth_imgs_path, crop_height=HEIGHT, crop_width=WIDTH, flag=False)
                gtImgTrain = np.zeros(GROUNDTRUTH_SHAPE_OR)
                gtImgTrain[0] = gt
                n_batches = int(len(training_imgs_path) // BATCH_SIZE)
                for batch in range(n_batches):
                    original_path = training_imgs_path[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE)]
                    original_batch = get_train_images(original_path, crop_height=HEIGHT, crop_width=WIDTH, flag=False)
                    original_batch = original_batch.reshape([BATCH_SIZE, 256, 256, 1])

                    ################################# TILE TRAINING CODE ##########################################

                    for tile_x in range(TILES_X):
                        for tile_y in range(TILES_Y):
                            x1 = tile_x*TRAIN_TAIL_SIZE_X
                            y1 = tile_y*TRAIN_TAIL_SIZE_Y
                            x2 = x1 + TRAIN_TAIL_SIZE_X
                            y2 = y1 + TRAIN_TAIL_SIZE_Y
                            tile_gt = gtImgTrain[0, x1: x2, y1: y2, 0]
                            tile_gt = tile_gt.reshape(GT_INPUT_SHAPE_TILE)
                            tile_Orig_Batch = []
                            for b in range(BATCH_SIZE):
                                tile_Orig_Batch.append(original_batch[b][x1:x2, y1: y2])
                            batchTile = np.stack(tile_Orig_Batch, axis=-1)
                            batchTile = batchTile.reshape([BATCH_SIZE, TRAIN_TAIL_SIZE_X, TRAIN_TAIL_SIZE_X, 1])
                            sess.run(train_op, feed_dict={original: batchTile, groundtruth: tile_gt})

                if debug:
                    for batch in range(n_batches):
                        original_path = training_imgs_path[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE)]
                        original_batch = get_train_images(original_path, crop_height=HEIGHT, crop_width=WIDTH, flag=False)
                        original_batch = original_batch.reshape([BATCH_SIZE, 256, 256, 1])

                        # print('original_batch shape final:', original_batch.shape)

                        # run the training step

                        ################################# TILE TRAINING CODE ##########################################
                        _p_loss = 0
                        for tile_x in range(TILES_X):
                            for tile_y in range(TILES_Y):
                                x1 = tile_x * TRAIN_TAIL_SIZE_X
                                y1 = tile_y * TRAIN_TAIL_SIZE_Y
                                x2 = x1 + TRAIN_TAIL_SIZE_X
                                y2 = y1 + TRAIN_TAIL_SIZE_Y
                                tile_gt = gtImgTrain[0, x1: x2, y1: y2, 0]
                                tile_gt = tile_gt.reshape(GT_INPUT_SHAPE_TILE)
                                tile_Orig_Batch = []
                                for b in range(BATCH_SIZE):
                                    tile_Orig_Batch.append(original_batch[b][x1:x2, y1: y2])
                                batchTile = np.stack(tile_Orig_Batch, axis=-1)
                                batchTile = batchTile.reshape([BATCH_SIZE, TRAIN_TAIL_SIZE_X, TRAIN_TAIL_SIZE_X, 1])
                                _p_loss += sess.run(pixel_loss, feed_dict={original: batchTile, groundtruth: tile_gt})
                        # add text file to add  mode(validation/training), epoch#, site#, batch#, _p_loss) ------------------------------
                        file.write('Train[Epoch#: %d, Site#: %d, Batch#: %d, _p_loss: %d]\n' % (epoch, site, batch, _p_loss))
                        print('Train[Epoch#: %d, Site#: %d, Batch#: %d, _p_loss: %d]' % (epoch, site, batch, _p_loss))
                print('Time taken per site: %s' %(datetime.now() - start_time_site))
                file.close()
            for site in range(numValSite):
                file = open(path + fileName, 'a')
                start_time_validation = datetime.now()
                groundtruth_val_imgs_path = list_images(validationPath+valFolders[site] + '/gt/')
                validation_imgs_path = list_images(validationPath+valFolders[site])
                np.random.shuffle(validation_imgs_path)
                gt = get_train_images(groundtruth_val_imgs_path, crop_height=HEIGHT, crop_width=WIDTH, flag=False)
                gtImgVal = np.zeros(GROUNDTRUTH_SHAPE_OR)
                gtImgVal[0] = gt
                val_batches = int(len(validation_imgs_path) // BATCH_SIZE)
                val_pixel_acc = 0
                for batch in range(val_batches):
                    val_original_path = validation_imgs_path[batch * BATCH_SIZE:(batch * BATCH_SIZE + BATCH_SIZE)]
                    val_original_batch = get_train_images(val_original_path, crop_height=HEIGHT, crop_width=WIDTH, flag=False)
                    val_original_batch = original_batch.reshape([BATCH_SIZE, 256, 256, 1])

                    ################################# TILE TRAINING CODE ##########################################
                    val_pixel = 0
                    for tile_x in range(TILES_X):
                        for tile_y in range(TILES_Y):
                            x1 = tile_x * TRAIN_TAIL_SIZE_X
                            y1 = tile_y * TRAIN_TAIL_SIZE_Y
                            x2 = x1 + TRAIN_TAIL_SIZE_X
                            y2 = y1 + TRAIN_TAIL_SIZE_Y
                            tile_gt = gtImgTrain[0, x1: x2, y1: y2, 0]
                            tile_gt = tile_gt.reshape(GT_INPUT_SHAPE_TILE)
                            tile_Orig_Batch = []
                            for b in range(BATCH_SIZE):
                                tile_Orig_Batch.append(val_original_batch[b][ x1:x2, y1:y2])
                            batchTile = np.stack(tile_Orig_Batch, axis=-1)
                            batchTile = batchTile.reshape([BATCH_SIZE, TRAIN_TAIL_SIZE_X, TRAIN_TAIL_SIZE_X, 1])
                            val_pixel += sess.run(pixel_loss, feed_dict={original: batchTile, groundtruth: tile_gt})
                    file.write('Validation[Epoch#: %d, Site#: %d, Batch#: %d, _p_loss: %d]\n' % (epoch, site, batch, val_pixel))
                    val_pixel_acc = val_pixel_acc + val_pixel
                print('Time taken per validation site: %s' % (datetime.now() - start_time_validation))
                val_loss = val_pixel_acc/val_batches
                file.write('ValidationAcc[Epoch#: %d, Site#: %d, Batch#: %d, val_loss: %d]\n' % (epoch, site, batch, val_loss))
                print('ValidationAcc[Epoch#: %d, Site#: %d, Batch#: %d, _p_loss: %d]' % (epoch, site, batch, val_loss))
                file.close()
            print('------------------------------------------------------------------------------')
            print('Time taken per epoc: %s' % (datetime.now() - start_time_epoc))
            saver.save(sess, save_path_epoc)
        saver.save(sess, save_path)
        print('Done training!')
        print('Total Time taken (training): %s' % (datetime.now() - start_time))
        file.close()
    # ** Done Training & Save the model **

def cropImage(img, x1, y1, x2, y2):
    tile = img.crop((x1, y1, x2, y2))
    return tile