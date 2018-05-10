# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 18:52:20 2017

@author: ybarancan
"""

import scipy.io as sio
import os
import glob
import h5py
import random
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

def valid_preprocess(path):
 
    
    mat_file = sio.loadmat(path)
    input_ = mat_file["input_"]
    label_ = mat_file["label_"]
    
    return input_/4095, label_/4095

def preprocess(path):
 
    
    mat_file = sio.loadmat(path)
    input_ = mat_file["input_"]
    label_ = mat_file["label_"]

    inputs, labels = expand(input_/4095,label_/4095)

    
    return inputs, labels


def read_data(path):

  with h5py.File(path, 'r') as hf:
    data = np.array(hf.get('data'))
    label = np.array(hf.get('label'))
    return data, label

def prepare_data(dataset):

  data = sorted(glob.glob(os.path.join(dataset, "*.mat")))

  print("Number of examples: "+ str(len(data)))
  return data


def prepare_data_test(dataset1, dataset2):

  data = sorted(glob.glob(os.path.join(dataset1, "*scale1.mat")))

  print("Number of examples: "+ str(len(data)))
  data2 =  sorted(glob.glob(os.path.join(dataset2, "*scale1.mat")))
  print("Number of examples: "+ str(len(data+data2)))
  return data

def prepare_data_valid(dataset):

  data = sorted(glob.glob(os.path.join(dataset, "*.mat")))

  print("Number of examples: "+ str(len(data)))
  return data

def make_data(sess, checkpoint_dir, data, label, batch_id=-1):

  if batch_id == -1:
      if FLAGS.train:
        savepath =  '{}/train.h5'.format(checkpoint_dir)
      else:
        savepath =  '{}/test.h5'.format(checkpoint_dir)
    
      with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('label', data=label)

  else:
      if FLAGS.train:
        savepath =  '{}/train_{}.h5'.format(checkpoint_dir,"batch_"+str(batch_id))
      else:
        savepath = '{}/test.h5'.format(checkpoint_dir)
    
      with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('label', data=label)



def batch_train_input_setup(config):

  
  sess = config.sess
  image_size, label_size, stride, scale = config.image_size, config.label_size, config.stride, config.scale

  print("image_size: "+str(image_size) + ", label size: "+str(label_size)+", stride:"+str(stride)+", scale: "+str(scale))
  # Load data path
  print(config.data_dir)

  data = prepare_data(sess, dataset=config.expand_images_dir)
  print("Only train folder : " + str(len(data)))



  sub_input_sequence, sub_label_sequence = [], []
  
  image_side = int(image_size/2)
  label_side = int(label_size/2)
  pad = config.pad
  
  random.shuffle(data)
  batch_id = 0
  batch_size = FLAGS.patch_batch_size
  counter = 0
  for i in range(len(data)):
    inputs, labels = preprocess(data[i])

    print("data : " + str(i))
    for m in range(8):
        
        input_ = inputs[m]
        label_ = labels[m]

        h, w, _ = input_.shape
    
        for x in range(pad + image_side, h - image_side - pad + 1, stride):
          for y in range(pad + image_side, w - image_side - pad  + 1, stride):
            sub_input = input_[x - pad - image_side  : x  + image_side + pad, y - pad -image_side : y  + image_side+pad, :]
           
            sub_label = label_[x - label_side : x + label_side, y -label_side : y + label_side, :]
            
            sub_input_sequence.append(sub_input)
            sub_label_sequence.append(sub_label)
            counter = counter + 1

            if ((counter % (400*batch_size)) == 0):
                
                indices = np.arange(len(sub_input_sequence))
                random.shuffle(indices)
                
                arrdata = np.asarray(sub_input_sequence)
                sub_input_sequence = []
                arrdata=arrdata[indices]
                
                arrlabel = np.asarray(sub_label_sequence)
                sub_label_sequence = []
                arrlabel=arrlabel[indices]
                
                for k in range(400):

                    make_data(sess, config.train_batches_dir, arrdata[k*batch_size:(k+1)*batch_size],
                              arrlabel[k*batch_size:(k+1)*batch_size], str(batch_id))
                    batch_id = batch_id + 1
                    print("Saved batch: "+str(batch_id))
                arrlabel = None
                arrdata=None
            
            
        

  if len(sub_input_sequence) != 0:   
    loops = counter % batch_size
    indices = np.arange(len(sub_input_sequence))
    random.shuffle(indices)
    arrdata = np.asarray(sub_input_sequence)[indices]
    arrlabel = np.asarray(sub_label_sequence)[indices]
    for k in range(loops):
        
        make_data(sess, config.train_batches_dir, arrdata[k*batch_size:(k+1)*batch_size],
                  arrlabel[k*batch_size:(k+1)*batch_size], str(batch_id))
        batch_id = batch_id + 1
        print("Saved batch: "+str(batch_id))
  print("All batches are saved")
  


def get_valid_data(config):
  
  
  data = prepare_data_valid(dataset=config.validation_dir)
  return data

def get_test_data(config):
  
  data = prepare_data_test(config.test_dir, config.test2_dir)
  return data

def get_valid_arrays(config):

  data = prepare_data(dataset=config.validation_dir)


 
  inputs = []
  labels = []
  for i in range(len(data)):
      input_, label_ = valid_preprocess(data[i])
      labels.append(label_)
      inputs.append(input_)
#    
#    
  return inputs, labels


def get_test_images(sess,test_path, scale):

  # Load data path
  data = prepare_data(sess, dataset=test_path)


  inputs = []
  labels = []
  for i in range(len(data)):
      input_, label_ = preprocess(data[i], scale, sess)
      labels.append(label_)
      inputs.append(input_)


  return inputs, labels


#////////////////////////////////////////////////////////////

def expand(input_,label_):
    
      out_in = []
      out_label = []
    
      for k in range(4):
          out_in.append(np.rot90(input_,k,(0,1)))
     
      for m in range(2):
          out_in.append(np.fliplr(out_in[m]))
          out_in.append(np.flipud(out_in[m]))

      for k in range(4):
          out_label.append(np.rot90(label_,k,(0,1)))
     
      for m in range(2):
          out_label.append(np.fliplr(out_label[m]))
          out_label.append(np.flipud(out_label[m]))
    
      return out_in, out_label

