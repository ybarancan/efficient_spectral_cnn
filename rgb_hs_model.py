# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 14:36:18 2017

@author: ybara
"""

from rgb_hs_data_prepare import (
  read_data, 
  batch_train_input_setup,
  get_valid_data,
  get_test_data,
  get_valid_arrays
)
import pprint
#from pre_trained import pre_trained_fsrcnn_inference
        
import scipy.io as sio
import glob
import random
import time
import os
import sys

import numpy as np
import tensorflow as tf

from rgb_hs_resnet import *

from math import log10

class Messenger_Obj(object):
    def __init__(self, batch_size, image_size, label_size, c_dim, pad):
        self.batch_size = batch_size
        self.image_size = image_size
        self.label_size = label_size
        self.c_dim = c_dim
        self.pad = pad
# Based on http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html
class spectral_model(object):
  
  def __init__(self, sess, config):
    self.sess = sess
    self.c_dim = config.c_dim
    self.epoch = config.epoch
    self.stride = config.stride
    self.batch_size = config.batch_size
    self.learning_rate = config.learning_rate
    self.momentum = config.momentum
    

    self.expand = config.expand
    self.load_checkpoint = config.loadCheckPoint
    self.train_batches_dir = config.train_batches_dir
  

    self.skip_data_preparation = config.skip_data_preparation

    self.validation_dir = config.validation_dir
    
    self.train_log_file = config.train_log_file
    self.vali_log_file = config.vali_log_file
    self.test_log_file = config.test_log_file
    self.patch_batch_size = config.patch_batch_size
    
    self.expand_images_dir = config.expand_images_dir
    self.create_expand_images = config.create_expand_images
    
    self.pad = 8
    
    self.image_size, self.label_size = [20,20] 
    self.data_image_size = self.image_size + 2*self.pad


    self.checkpoint_dir = config.checkpoint_dir
    self.best_checkpoint_dir = config.best_checkpoint_dir
    self.output_dir = config.output_dir
    self.load_checkpoint_dir = config.load_checkpoint_dir
    self.test_dir = config.test_dir
    self.test2_dir = config.test2_dir
    self.scrap_dir = config.scrap_dir
    self.pp = pprint.PrettyPrinter()
    self.build_model()


  def build_model(self):
    self.images = tf.placeholder(tf.float32, [None, None, None, 3], name='images')
    self.labels = tf.placeholder(tf.float32, [None, None, None, self.c_dim], name='labels')
    

    self.batch = tf.placeholder(tf.int32, shape=[], name='batch')
    
    self.whole_image = tf.placeholder(tf.float32, [1, None, None,3], name='whole_image')   
    
    obj = Messenger_Obj(self.batch_size,self.data_image_size, self.label_size,self.c_dim,self.pad)
    

    self.pred = residual_hyper_inference(self.images, False, self.batch,obj)

    self.loss = tf.reduce_sum(tf.pow(self.pred - self.labels, 2))/(self.batch_size*self.label_size*self.label_size*self.c_dim*2) 

    self.saver = tf.train.Saver()

  def run(self):
      
    
    global_step = tf.Variable(0, trainable=False)
    self.starter_learning_rate = np.copy(self.learning_rate)
    self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, global_step,
                                               50000, 0.93, staircase=True)

    self.train_op = tf.train.AdamOptimizer(self.learning_rate,epsilon=1e-6).minimize(self.loss, global_step
                                             = global_step)

    tf.initialize_all_variables().run()
    
    self.run_batch_train()

  def run_batch_train(self):
    start_time = time.time()
    print("Beginning batch training setup...")


    if not self.skip_data_preparation:
        batch_train_input_setup(self)

    print("Training setup took {} seconds w".format(time.time() - start_time))
#
    valid_datas = get_valid_data(self)
    test_datas = get_test_data(self)
    n_vali = len(valid_datas)
    n_test = len(test_datas)
    if self.load_checkpoint:
        self.load(self.load_checkpoint_dir)
        
    counter = 0
    losses = np.zeros((1000))


    best_rgb_val_loss = (10000,0)
    best_rgb_test_loss = (10000,0)
    print("Training...")
    start_time = time.time()

    batches = sorted(glob.glob(os.path.join(self.train_batches_dir,"*.h5")))
    print("Len of batches: "+str(len(batches)))


    for ep in range(self.epoch):

      batch_shuffled_id = np.arange(len(batches))
      random.shuffle(batch_shuffled_id)

      for batch_id in range(len(batches)):
        
        data_dir = batches[batch_shuffled_id[batch_id]]
    
        train_data, train_label = read_data(data_dir)
        tempind = np.arange(len(train_data))
        random.shuffle(tempind)

        batch_idxs = len(train_data) // self.batch_size
        

        
        batch_average = 0
        for idx in range(0, batch_idxs):

            batch_images = [train_data[i] for i in tempind[idx * self.batch_size : (idx + 1) * self.batch_size]]
            batch_labels = [train_label[i] for i in tempind[idx * self.batch_size : (idx + 1) * self.batch_size]]
        
            counter += 1
            
            
            _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels, self.batch: len(batch_images)})
            
            
            batch_average += err

                
            if counter % 50000 == 0:
                self.save(self.checkpoint_dir, counter)
            losses[counter % 1000] = err
            
#            VALIDATION


            if (counter % 50000 == 0) :
             
              GRMSEs=[]
              GrRMSEs=[]
              ARMSEs = []
              ArRMSEs = []
              
              uGRMSEs=[]
              uGrRMSEs=[]
              uARMSEs = []
              uArRMSEs = []
              
              for k in range(len(valid_datas)):
                  
                  vali_in, vali_la = self.mat_preprocess(valid_datas[k])
                  test_label = vali_la 
                  test_label = test_label[self.pad:test_label.shape[0]-self.pad,self.pad:test_label.shape[1]-self.pad, : ]         
                  for m in range(8):
                      
                      """
                      In order not to have problems with RAM, the images are divided to 4 sub images during test time with counstruct and deconstruct
                      You can uncomment the following segment and comment the part after it to not use dividing to sub images.Especially if the images you
                      are using are small.
                      """
#                      test_data = self.single_geo_preprocess(vali_in,m)
# 
#                      result = self.pred.eval(feed_dict={self.images: np.reshape(test_data, (1,test_data.shape[0],test_data.shape[1],3)),self.batch: 1})
#                 
#                      result = result.squeeze()
#                      result[result<0] = 0
#                      result[result > 1] = 1
#                      te_result = self.single_geo_postprocess(result,m)
#                      if m == 0:
#                          fin_res = te_result
#                      else:
#                          fin_res = fin_res + te_result
                          
                      test_data = self.single_geo_preprocess(vali_in,m)
                      res_parts = []
                      for piece in range(4):
                          part = self.deconstruct(test_data,piece,self.pad)
                          res_part = self.pred.eval(feed_dict={self.images: np.reshape(part, (1,part.shape[0],part.shape[1],3)), self.batch: 1})
                          res_parts.append(res_part)
                      
                      result = self.construct(res_parts,(test_data.shape[0]-2*self.pad,test_data.shape[1]-2*self.pad,31))
                      result = result.squeeze()
                      result[result<0] = 0
                      result[result > 1] = 1
                      te_result = self.single_geo_postprocess(result,m)
                      if m == 0:
                          fin_res = te_result
                      else:
                          fin_res = fin_res + te_result
                          
# / //////////////////////////////////////////   
                       
                  temp = test_label.squeeze()
                  temp_result = (fin_res)/8
                  temp_temp = (temp)
                
                  grmse, grrmse, armse,arrmse = self.psnr(temp_result,temp_temp , False)
                  GRMSEs.append(grmse)
                  GrRMSEs.append(grrmse)
                  ARMSEs.append(armse)
                  ArRMSEs.append(arrmse)
                  
                  ugrmse, ugrrmse, uarmse,uarrmse = self.psnr(temp_result,temp_temp , True)
                  uGRMSEs.append(ugrmse)
                  uGrRMSEs.append(ugrrmse)
                  uARMSEs.append(uarmse)
                  uArRMSEs.append(uarrmse)
                  
                  text_file = open("individual_vali.txt", "a")
              
                  text_file.write("\nGaliani RMSE " + " : "+str(k) + " : "+str(grmse))
                  text_file.write("\nGaliani RRMSE" + " : "+str(k) + " : "+str(grrmse))
                  text_file.write("\nArad RMSE " + " : "+str(k) + " : "+str(armse))
                  text_file.write("\nArad RRMSE : " +" : " + str(k) + " : "+str(arrmse))
                  
                  text_file.write("\nGaliani uint RMSE " + " : "+str(k) + " : "+str(ugrmse))
                  text_file.write("\nGaliani uint RRMSE" + " : "+str(k) + " : "+str(ugrrmse))
                  text_file.write("\nArad uint RMSE " + " : "+str(k) + " : "+str(uarmse))
                  text_file.write("\nArad uint RRMSE : " +" : " + str(k) + " : "+str(uarrmse))
                  text_file.close()   
                  
                  print("Validated image : " + str(k))


              temp_float = sum(GRMSEs)/n_vali
              print("Current RMSE: " + str(temp_float))
              print("RMSE best : " + str(best_rgb_val_loss[0]))
              if  temp_float < best_rgb_val_loss[0]:
                  best_rgb_val_loss = (temp_float, counter) 
                 
                  self.save(self.best_checkpoint_dir, counter)
                  
              text_file = open(self.vali_log_file, "a")
              
              text_file.write("\nAverage Galiani RMSE " + " : "+str(counter) + " : "+str(sum(GRMSEs)/len(valid_datas)))
              text_file.write("\nAverage Galiani RRMSE" + " : "+str(counter) + " : "+str(sum(GrRMSEs)/len(valid_datas)))
              text_file.write("\nAverage Arad RMSE " + " : "+str(counter) + " : "+str(sum(ARMSEs)/len(valid_datas)))
              text_file.write("\nAverage Arad RRMSE : " +" : " + str(counter)+":" + str(sum(ArRMSEs)/len(valid_datas)))
              
              text_file.write("\nAverage Galiani uint RMSE " + " : "+str(counter) + " : "+str(sum(uGRMSEs)/len(valid_datas)))
              text_file.write("\nAverage Galiani uint RRMSE" + " : "+str(counter) + " : "+str(sum(uGrRMSEs)/len(valid_datas)))
              text_file.write("\nAverage Arad uint RMSE " + " : "+str(counter) + " : "+str(sum(uARMSEs)/len(valid_datas)))
              text_file.write("\nAverage Arad uint RRMSE : " +" : " + str(counter)+":" + str(sum(uArRMSEs)/len(valid_datas)))
              
              text_file.write("\nMax RMSE : " + str(best_rgb_val_loss[1]) +" : " + str(best_rgb_val_loss[0]))
              
              text_file.close()    

              print("\nAverage PSNR VAlidation : " + str(counter) +" : " + str(sum(GRMSEs)/len(valid_datas)))


            if counter % 1000 == 0:
              text_file = open(self.train_log_file, "a")
              text_file.write("\nError in step " + str(counter) +" : " + str(sum(losses)/1000))
              text_file.close()
              print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
                % ((ep+1), counter, time.time() - start_time, sum(losses)/1000))
              

  def geo_preprocess(self,data):
      out_data = []
    
      for k in range(4):
          out_data.append(np.rot90(data,k,(0,1)))
     
      for m in range(2):
          out_data.append(np.fliplr(out_data[m]))
          out_data.append(np.flipud(out_data[m]))
    
      return out_data
  
  def geo_postprocess(self,data):
      out_data = []
      for k in range(len(data)):
          if k < 4 : 
              out_data.append(np.rot90(data[k],4-k,(0,1)))
          else:
              if k == 4:
                  out_data.append(np.fliplr(data[k]))
              elif k==5:
                  out_data.append(np.flipud(data[k]))
              elif k==6:
                  out_data.append(np.rot90(np.fliplr(data[k]),3,(0,1)))
              else:
                  out_data.append(np.rot90(np.flipud(data[k]),3,(0,1)))
    
    
      return np.mean(np.asarray(out_data),0)
    
  def single_geo_preprocess(self,data,n):
   
      if n < 4:
          return np.rot90(data,n,(0,1))
      elif n == 4:
          return np.fliplr(data)
      elif n == 5:
          return np.flipud(data)
      elif n == 6:
          return np.fliplr(np.rot90(data,1,(0,1)))
      elif n == 7:
          return np.flipud(np.rot90(data,1,(0,1)))

      else:
          return None
  
  def single_geo_postprocess(self,data,n):
      
      if n < 4:
          return np.rot90(data,4-n,(0,1))
      elif n == 4:
          return np.fliplr(data)
      elif n == 5:
          return np.flipud(data)
      elif n == 6:
          return (np.rot90(np.fliplr(data),3,(0,1)))
      elif n == 7:
          return (np.rot90(np.flipud(data),3,(0,1)))

      else:
          return None
      
        
  def deconstruct(self,data,piece,pad):
      
      width = data.shape[0]
      height = data.shape[1]
      if piece == 0:
          return data[0:int(width/2 + pad), 0:int(height/2 + pad),:]
      if piece == 1:
          return data[int(width/2 - pad):, 0:int(height/2 + pad),:]
      if piece == 2:
          return data[0:int(width/2 + pad), int(height/2 - pad):,:]
      if piece == 3:
          return data[int(width/2 - pad):, int(height/2 - pad):,:]
    
  def construct(self, datas,shape):
      
      width = shape[0]
      height = shape[1]
      res = np.zeros(shape)
      res[0:int(width/2) , 0:int(height/2) ,:] = datas[0]
      res[int(width/2):, 0:int(height/2) ,:] = datas[1]
      res[0:int(width/2) , int(height/2):,:] = datas[2]
      res[int(width/2):, int(height/2):,:] = datas[3]
      
      return res

  def save(self, checkpoint_dir, step):
    model_name = "spectral.model"

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        print(str(ckpt) + " and : " + str(ckpt_name))
        return True
    else:
        return False
    
  def psnr(self,img1, img2, unt):
    print("img1 max : "+str(max(img1.flatten())) + " : mean : "+ str(np.mean(img1)) )
    print("img2 max : "+str(max(img2.flatten()))+ " : mean : "+ str(np.mean(img2)) )
    scale = 2
    if unt:
        
        img1 = (255*img1).astype(np.uint8).squeeze()
        img2 = (255*img2).astype(np.uint8).squeeze()
        PIXEL_MAX = 255
    else:
        img1 = img1.astype(np.float32).squeeze()
        img2 = img2.astype(np.float32).squeeze()
        img2[img2 == 0] = 0.0001
        PIXEL_MAX = 1
    if len(img1.shape) == 2:
        img1 = img1[scale:-scale,scale:-scale]
        img2 = img2[scale:-scale,scale:-scale]
    else:
        img1 = img1[scale:-scale,scale:-scale,:]
        img2 = img2[scale:-scale,scale:-scale,:]
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    
    t = np.sqrt(mse)
    
    aae = np.mean(np.sqrt((img1 - img2) ** 2))
    temp = np.mean(np.sqrt((img1 - img2) ** 2)/img2)
    
    return t, t/np.mean(img2),aae, temp
#    
#    return 10 * log10(PIXEL_MAX*PIXEL_MAX / (mse))


  def mat_preprocess(self,path):
 
    
    mat_file = sio.loadmat(path)
    input_ = mat_file["input_"]/4095
    label_ = mat_file["label_"]/4095
#    print("Length 1: " + str(input_.shape))
    return input_, label_
