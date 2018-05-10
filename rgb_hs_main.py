# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 10:55:05 2017

@author: ybarancan
"""

from rgb_hs_model import spectral_model


import tensorflow as tf

import pprint
import os

gpu =1



flags = tf.app.flags

flags.DEFINE_integer("epoch", 1000000, "Number of epochs (Not really used)")
flags.DEFINE_integer("batch_size", 64, "Number of images in a batch")
flags.DEFINE_float("learning_rate", 0.5*(1e-3), "The initial learning rate for Adam Optimizer")
flags.DEFINE_float("momentum", 0.9, "The momentum value for Adam Optimizer (Not used as default)")
flags.DEFINE_integer("c_dim", 31, "Dimension of hyperspectral image")
flags.DEFINE_integer("stride", 32, "The stride to extract patches from training images")
flags.DEFINE_string("checkpoint_dir", "check", "Name of regular checkpoint directory")
flags.DEFINE_string("best_checkpoint_dir", "/scratch/cany/best_checkpoint/"+ str(gpu)+"/", "Name of checkpoint directory in which whenever best validation error is achieved")
flags.DEFINE_string("output_dir", "/scratch/cany/result_pics/"+str(gpu)+"/", "Name of output directory (Not used in default)")

flags.DEFINE_boolean("expand", True, "Expand the dataset by rotating and scaling")
flags.DEFINE_boolean("create_expand_images", False, "Use Matlab to imresize the train images and put them in expanded_images_dir. The rotation and flipping will be done by this code during training")
flags.DEFINE_string("expand_images_dir", "/scratch/cany/icvl/expanded/set"+str(gpu)+"/train/", "Name of the directory to put the expanded data in")
flags.DEFINE_string("scrap_dir", "/scratch/cany/icvl/expanded/set"+str(gpu)+"/scrap/", "Name of the directory that expanded validation data resides")

flags.DEFINE_boolean("loadCheckPoint", True, "Load the model from the checkpoint")


flags.DEFINE_string("train_batches_dir", "/scratch/cany/icvl/expanded/set"+str(gpu)+"/batches/", "Name of directory in which train batches reside ")


flags.DEFINE_boolean("skip_data_preparation", True, "Skip preparing the batches if they are already there")

flags.DEFINE_string("validation_dir", "/scratch/cany/icvl/expanded/set"+str(gpu)+"/validation/", "Name of directory in which validation images reside ")

flags.DEFINE_string("test_dir", "/scratch/cany/icvl/expanded/set"+str(1-gpu)+"/train/", "Name of directory in which test images reside ")
flags.DEFINE_string("test2_dir", "/scratch/cany/icvl/expanded/set"+str(1-gpu)+"/validation/", "Name of directory in which test images reside ")

flags.DEFINE_string("train_log_file", "train_error_log.txt", "Name of txt file that current training loss will be written ")
flags.DEFINE_string("vali_log_file", "vali_error_log.txt", "Name of txt file that current validation loss will be written ")
flags.DEFINE_string("test_log_file", "test_error_log.txt", "Name of txt file that current test loss will be written ")

flags.DEFINE_integer("patch_batch_size", 1280, "How many patches to put in a batch save. This is not the size of batches in training!!!")

flags.DEFINE_string("load_checkpoint_dir", "check", "Name of directory in which checkpoint to load resides ")


FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):
  pp.pprint(flags.FLAGS.__flags)
  os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.best_checkpoint_dir):
    os.makedirs(FLAGS.best_checkpoint_dir)
  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)
  if not os.path.exists(FLAGS.train_batches_dir):
    os.makedirs(FLAGS.train_batches_dir)

  with tf.Session(config = config) as sess:
    model = spectral_model(sess, config=FLAGS)
    model.run()
    
if __name__ == '__main__':
  tf.app.run()
