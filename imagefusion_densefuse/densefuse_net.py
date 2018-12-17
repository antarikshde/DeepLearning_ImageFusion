# DenseFuse Network
# Encoder -> Addition/L1-norm -> Decoder

import tensorflow as tf

from encoder import Encoder
from decoder import Decoder
from fusion_addition import Strategy

class DenseFuseNet(object):

    def __init__(self, model_pre_path):
        self.encoder = Encoder(model_pre_path)
        self.decoder = Decoder(model_pre_path)

    def transform_addition(self, imgs):
        # encode image
        size = len(imgs)
        encs = ["" for x in range(size)]
        encs[0] = self.encoder.encode(imgs[0])
        encs[1] = self.encoder.encode(imgs[1])
        target_features = Strategy(encs[0], encs[1])
        for x in range(2, size):
            encs[x] = self.encoder.encode(imgs[x])
            target_features = Strategy(target_features, encs[x])
        self.target_features = target_features
        print('target_features:', target_features.shape)
        # decode target features back to image
        generated_img = self.decoder.decode(target_features)
        return generated_img

    def transform_addition_train(self, imgs):
        # encode image
        size = len(imgs)
        target_features = Strategy(imgs[0], imgs[1])
        for x in range(2, size):
            target_features = Strategy(target_features, imgs[x])
        self.target_features_train = target_features
        print('target_features_train.shape:', target_features.shape)
        return target_features

    def transform_recons(self, img):
        # encode image
        enc = self.encoder.encode(img)
        target_features = enc
        self.target_features = target_features
        generated_img = self.decoder.decode(target_features)
        return generated_img

    def transform_recons_train(self, img):
        # encode image
        enc = self.encoder.encode(img)
        target_features = enc
        print('target_features.shape: ', target_features.shape)
        target1, target2, target3, target4 = tf.split(target_features, 4, 0)
        print('target1.shape: ', target1.shape)
        individual_target_features = [target1, target2, target3, target4]
        final_target_features = self.transform_addition_train(individual_target_features)
        print('final_target_features.shape: ', final_target_features.shape)
        self.target_features_train = final_target_features
        generated_img = self.decoder.decode(final_target_features)
        return generated_img

    def transform_encoder(self, imgs):
        # encode image
        size = len(imgs)
        encs = ["" for x in range(size)]
        for x in range(0, size):
            encs[x] = self.encoder.encode(imgs[x])
        return encs

    def transform_decoder(self, feature):
        # decode image
        generated_img = self.decoder.decode(feature)
        return generated_img

