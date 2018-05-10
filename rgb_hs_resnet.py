
import numpy as np
import tensorflow as tf
import scipy.io as sio


def prelu( _x, i, reuse = None, custom_init = False, force=None):

    if force != None:
        shap = force
    else:
        shap=_x.get_shape()[-1]
    if custom_init:
        if reuse != None:
            alphas = tf.get_variable('alpha{}'.format(i), shap, initializer=tf.constant_initializer(reuse), dtype=tf.float32)
    else:
        if reuse != None:
            alphas = reuse
        else:
            alphas = tf.get_variable('alpha{}'.format(i),shap, initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg


def residual_hyper_inference(input_tensor_batch,  reuse, batch,config):

    pad = config.pad
    deconv_width = 7
    deconv_one = int(deconv_width/2)
    rgb = 3
    c_dim = config.c_dim
    
#    FSRCNN notation. You can take a look at FSRCNN paper for a deeper explanation
    s, d, m = (128,32,4)

    expand_weight, deconv_weight, upsample_weight = 'w{}'.format(m + 3), 'w{}'.format(m + 4),'w{}'.format(m + 5) 
    weights = {
      'w1': tf.get_variable('w1', shape=[5,5,rgb,s], initializer=tf.contrib.layers.xavier_initializer()),
      'w2': tf.get_variable('w2', shape=[1,1,s,d], initializer=tf.contrib.layers.xavier_initializer()),
      expand_weight: tf.get_variable(expand_weight, shape=[1,1,d,s], initializer=tf.contrib.layers.xavier_initializer()),
      deconv_weight: tf.get_variable(deconv_weight, shape=[5,5,s,c_dim], initializer=tf.contrib.layers.xavier_initializer()),
      upsample_weight: tf.get_variable(upsample_weight,shape=[deconv_width, deconv_width, rgb, c_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
      
    }

    expand_bias, deconv_bias, upsample_bias = 'b{}'.format(m + 3), 'b{}'.format(m + 4), 'b{}'.format(m + 5)
    biases = {
      'b1': tf.Variable(tf.zeros([s]), name='b1'),
      'b2': tf.Variable(tf.zeros([d]), name='b2'),
      expand_bias: tf.Variable(tf.zeros([s]), name=expand_bias),
      deconv_bias: tf.Variable(tf.zeros([c_dim]), name=deconv_bias),
      upsample_bias: tf.Variable(tf.zeros([c_dim]), name=upsample_bias)
    }


    for i in range(3, m + 3):
      weight_name, bias_name = 'w{}'.format(i), 'b{}'.format(i)
      weights[weight_name] = tf.get_variable(weight_name, shape=[3,3,d,d], initializer=tf.contrib.layers.xavier_initializer())
      biases[bias_name] = tf.Variable(tf.zeros([d]), name=bias_name)
      
    conv_feature = prelu(tf.nn.conv2d(input_tensor_batch, weights['w1'], strides=[1,1,1,1], padding='VALID') + biases['b1'], 1)

    # Shrinking
    conv_shrink = prelu(tf.nn.conv2d(conv_feature, weights['w2'], strides=[1,1,1,1], padding='VALID') + biases['b2'], 2)

    prev_layer = conv_shrink 
    for k in range(3, m + 3):
      weight, bias = weights['w{}'.format(k)], biases['b{}'.format(k)]
      if k == 3:
          prev_layer = prelu(tf.nn.conv2d(prev_layer, weight, strides=[1,1,1,1], padding='VALID') + bias, k)
      if k == 4:
          prev_layer1 = prelu(tf.nn.conv2d(prev_layer, weight, strides=[1,1,1,1], padding='VALID') + bias+tf.slice(conv_shrink, [0,2,2, 0], [-1,tf.shape(conv_shrink)[1]-4,tf.shape(conv_shrink)[2]-4 ,-1]), k)
      if k==5:
          prev_layer = prelu(tf.nn.conv2d(prev_layer1, weight, strides=[1,1,1,1], padding='VALID') + bias, k)
      if k == 6:
          prev_layer = prelu(tf.nn.conv2d(prev_layer, weight, strides=[1,1,1,1], padding='VALID') + bias + 
                                          tf.slice(conv_shrink, [0,4,4,0],[-1,tf.shape(conv_shrink)[1]-8,tf.shape(conv_shrink)[2]-8,-1]) +
                                          tf.slice(prev_layer1, [0,2,2, 0], [-1,tf.shape(prev_layer1)[1]-4,tf.shape(prev_layer1)[2]-4 ,-1]), k)
    
    
    
    # Expanding
    expand_weights, expand_biases = weights['w{}'.format(m + 3)], biases['b{}'.format(m + 3)]
    conv_expand = prelu(tf.nn.conv2d(prev_layer, expand_weights, strides=[1,1,1,1], padding='VALID') + expand_biases + tf.slice(conv_feature, [0,4,4, 0], [-1,tf.shape(conv_feature)[1]-8,tf.shape(conv_feature)[2]-8 ,-1]), 7)
    
    
    primitive_upsampled = tf.nn.conv2d(input_tensor_batch, weights['w9'], strides=[1,1,1,1], padding='VALID') + biases['b9']

    net =  prelu(tf.nn.conv2d(conv_expand, weights['w8'], strides=[1,1,1,1], padding='VALID') + biases['b8']+ tf.slice(primitive_upsampled, [0,8-deconv_one,8-deconv_one, 0], [-1, tf.shape(input_tensor_batch)[1]-16,tf.shape(input_tensor_batch)[2]-16 ,-1]),8)
    

    return net


