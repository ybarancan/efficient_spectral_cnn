# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 22:55:29 2017

@author: ybarancan
"""

import scipy.io as sio
from rgb_hs_resnet import *

import glob
import random
import time
import os
import sys

import numpy as np
import tensorflow as tf




from math import log10

dataset = "/scratch/cany/icvl/expanded/set0/train/"
dataset2 = "/scratch/cany/icvl/expanded/set0/validation/"
gpu = 1
pad = 8
label_size = 20
image_size = 20
c_dim=31
data_image_size = image_size + 2*pad
output_dir = "output/"
model_dir = "check"

class Messenger_Obj(object):
    def __init__(self, sess, image_size, label_size,c_dim,pad):

        self.sess = sess
        self.c_dim= c_dim
        self.image_size = image_size
        self.label_size = label_size
        self.pad = pad
        

def prepare_data_test( dataset1, dataset2):

  data = sorted(glob.glob(os.path.join(dataset1, "*.mat")))

  return data
def load(sess, saver,  checkpoint_dir):
    print(" [*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    print(str(ckpt))
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False

def tester_psnr(img1, img2, unt):
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



def geo_preprocess(data):
  out_data = []

  for k in range(4):
      out_data.append(np.rot90(data,k,(0,1)))
 
  for m in range(2):
      out_data.append(np.fliplr(out_data[m]))
      out_data.append(np.flipud(out_data[m]))

  return out_data
  
def geo_postprocess(data):
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

def mat_preprocess(path):
 
    
    mat_file = sio.loadmat(path)
    input_ = mat_file["input_"]/4095
    label_ = mat_file["label_"]/4095
    
    return input_, label_

def load_conv_matrix(path):
    mat_file = sio.loadmat(path)
    input_ = mat_file["normalized_HStoRGB"]
    return input_

def single_geo_preprocess(data,n):
   
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
  
def single_geo_postprocess(data,n):
      
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
      
        
def deconstruct(data,piece,pad):
  
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

def construct( datas,shape):
  
  width = shape[0]
  height = shape[1]
  res = np.zeros(shape)
  res[0:int(width/2) , 0:int(height/2) ,:] = datas[0]
  res[int(width/2):, 0:int(height/2) ,:] = datas[1]
  res[0:int(width/2) , int(height/2):,:] = datas[2]
  res[int(width/2):, int(height/2):,:] = datas[3]
  
  return res
whole_image = tf.placeholder(tf.float32, [1, None, None, 3], name='whole_image')   

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

sess = tf.Session(config = config)


obj = Messenger_Obj(sess, 20, 20,c_dim,pad)

images = tf.placeholder(tf.float32, [None, data_image_size, data_image_size, 3], name='images')
labels = tf.placeholder(tf.float32, [None, label_size, label_size, c_dim], name='labels')

batch = tf.placeholder(tf.int32, shape=[], name='batch')

pred = residual_hyper_inference(whole_image,  False, batch,obj)

saver = tf.train.Saver()
load(sess,saver,checkpoint_dir=model_dir)

valid_datas = prepare_data_test(dataset,dataset2)


psnrs = []
float_psnrs=[]
nRMSEs = []
aaes = []
raaes = []

uint_rmses = []
uint_rrmses=[]
uint_arad=[]
uint_rarad=[]


with sess:
    
  for k in range(len(valid_datas)):
                  
      vali_in, vali_la = mat_preprocess(valid_datas[k])
      print("input max : "+str(max(vali_in.flatten())) + " : mean : "+ str(np.mean(vali_in)) )
      test_label = vali_la
      
      test_label = test_label[pad:test_label.shape[0]-pad,pad:test_label.shape[1]-pad, : ]
      
      s = time.time()
      for m in range(8):
    
          test_data = single_geo_preprocess(vali_in,m)
#          res_parts = []
#          for piece in range(4):
#              part = deconstruct(test_data,piece,pad)
#              res_part = pred.eval(feed_dict={whole_image: np.reshape(part, (1,part.shape[0],part.shape[1],3)), batch: 1})
#              res_parts.append(res_part)
#                      
#          result = construct(res_parts,(test_data.shape[0]-2*pad,test_data.shape[1]-2*pad,31))
#          s = time.time()
          result = pred.eval(feed_dict={whole_image: np.reshape(test_data, (1,test_data.shape[0],test_data.shape[1],3)),batch: 1})
         
          
          result = result.squeeze()
          result[result<0] = 0
          result[result > 1] = 1
          te_result = single_geo_postprocess(result,m)
          if m == 0:
              fin_res = te_result
          else:
              fin_res = fin_res + te_result
              
      temp = test_label.squeeze()


      temp_result = (fin_res)/8
      temp_temp = (temp)

      psnr_float, nRMSE, aae, raae = tester_psnr(temp_result,temp_temp , False)
      float_psnrs.append(psnr_float)
      nRMSEs.append(nRMSE)
      aaes.append(aae)
      raaes.append(raae)
      psnr_uint, rrmse_galiani, wh, ev = tester_psnr(temp_result,temp_temp , True)
      uint_rmses.append(psnr_uint)
      uint_rrmses.append(rrmse_galiani)
      uint_arad.append(wh)
      uint_rarad.append(ev)

      text_file = open("tester.txt", "a")
              
      text_file.write("\nGaliani RMSE of float " + " : "+str(k) + " : "+str(psnr_float))
      text_file.write("\nGaliani RRMSE " + " : "+str(k) + " : "+str(nRMSE))
      text_file.write("\nGaliani uint RMSE " + " : "+str(k) + " : "+str(psnr_uint))
      text_file.write("\nGaliani uint RRMSE " + " : "+str(k) + " : "+str(rrmse_galiani))
      text_file.write("\nArad RMSE " + " : "+str(k) + " : "+str(aae))
      text_file.write("\nArad RRMSE" + " : "+str(k) + " : "+str(raae))
      text_file.close()  
      print("\nGaliani RMSE of float " + " : "+str(k) + " : "+str(psnr_float))
      print("\nGaliani RRMSE " + " : "+str(k) + " : "+str(nRMSE))
      print("\nArad RMSE " + " : "+str(k) + " : "+str(aae))
      print("\nArad RRMSE" + " : "+str(k) + " : "+str(raae))
     
      


print("\nAverage Galiani RMSE : " +" : " + str(sum(float_psnrs)/len(valid_datas)))
print("\nAverage Galiani RRMSE : " +" : " + str(sum(nRMSEs)/len(valid_datas)))
print("\nAverage Arad RMSE : " +" : " + str(sum(aaes)/len(valid_datas)))
print("\nAverage Arad RRMSE : " +" : " + str(sum(raaes)/len(valid_datas)))


text_file = open("tester.txt", "a")
      
text_file.write("\nAverage Galiani RMSE " + " : "+str(k) + " : "+str(sum(float_psnrs)/len(valid_datas)))
text_file.write("\nAverage Galiani RRMSE" + " : "+str(k) + " : "+str(sum(nRMSEs)/len(valid_datas)))
text_file.write("\nAverage Arad RMSE " + " : "+str(k) + " : "+str(sum(aaes)/len(valid_datas)))
text_file.write("\nAverage Arad RRMSE : " +" : " + str(sum(raaes)/len(valid_datas)))

text_file.write("\nAverage Galiani uint RMSE " + " : "+str(sum(uint_rmses)/len(valid_datas)))
text_file.write("\nAverage Galiani uint RRMSE" + " : "+str(sum(uint_rrmses)/len(valid_datas)))
text_file.write("\nAverage Arad uint RMSE " + " : "+str(sum(uint_arad)/len(valid_datas)))
text_file.write("\nAverage Arad uint RRMSE" + " : "+str(sum(uint_rarad)/len(valid_datas)))

text_file.close()  
