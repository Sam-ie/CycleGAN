#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from random import shuffle
import numpy as np
import argparse
import tensorflow as tf
import random
import cv2
from numba import jit

# import ssim_exp
# from skimage.measure import _structural_similarity as ssim
# import function.ssim_multiscale as ssim
# import skimage.measure._structural_similarity

# image_1为原图，image_2为生成图片
# @jit
# def ssim(image_1,image_2,c_1=0.02,c_2=0.03,patch=3.,expand=1):##expand和patch是联动的
#    
#    img_x_1 =tf.Variable( tf.zeros((256+patch-1,256+patch-1)),name = 'img_x_1' )
#    img_x_2 =tf.Variable( tf.zeros((256+patch-1,256+patch-1)),name = 'img_x_2' )
#    img_x_3 =tf.Variable( tf.zeros((256+patch-1,256+patch-1)),name = 'img_x_3' )
#    img_y_1 =tf.Variable( tf.zeros((256+patch-1,256+patch-1)),name = 'img_y_1' )
#    img_y_2 =tf.Variable( tf.zeros((256+patch-1,256+patch-1)),name = 'img_y_2'  )
#    img_y_3 =tf.Variable( tf.zeros((256+patch-1,256+patch-1)),name = 'img_y_3'  )
#    
#    img_x_1 = tf.assign(img_x_1[expand:-expand,expand:-expand], image_1[0,:,:,0], validate_shape=True, use_locking=None, name=None)
#    img_x_2 = tf.assign(img_x_2[expand:-expand,expand:-expand], image_1[0,:,:,1], validate_shape=True, use_locking=None, name=None)
#    img_x_3 = tf.assign(img_x_3[expand:-expand,expand:-expand], image_1[0,:,:,2], validate_shape=True, use_locking=None, name=None)
#    
#    img_y_1 = tf.assign(img_y_1[expand:-expand,expand:-expand], image_2[0,:,:,0], validate_shape=True, use_locking=None, name=None)
#    img_y_2 = tf.assign(img_y_2[expand:-expand,expand:-expand], image_2[0,:,:,1], validate_shape=True, use_locking=None, name=None)
#    img_y_3 = tf.assign(img_y_3[expand:-expand,expand:-expand], image_2[0,:,:,2], validate_shape=True, use_locking=None, name=None)
#    
#    ssim_loss_1 = tf.Variable( 0,name = 'ssim_loss_1' )
#    for i in range(256):
#        for j in range(256):
#            u_x_1 = tf.reduce_mean(img_x_1[i:i+patch,j:j+patch])
#            u_y_1 = tf.reduce_mean(img_y_1[i:i+patch,j:j+patch])
#            v_x_1 = tf.reduce_mean((img_x_1[i:i+patch,j:j+patch]-u_x_1)**2)
#            v_y_1 = tf.reduce_mean((img_y_1[i:i+patch,j:j+patch]-u_y_1)**2)
#            c_v_1 = tf.reduce_mean( (img_y_1[i:i+patch,j:j+patch]-u_y_1)*(img_x_1[i:i+patch,j:j+patch]-u_x_1) )
#    return img_x_1
# image_1 = cv2.imread('E:\\transfer_learning\\image_sets\\type_1\\image_1.jpg')
# image_2 = cv2.imread('E:\\transfer_learning\\image_sets\\type_1\\image_2.jpg')
# index = ssim.compare_ssim(image_1,image_2,win_size = 13,multichannel = True,gaussian_weights=True)
# djj_1 = np.ones((1,2000,1504,3))
# djj_2 = np.ones((1,2000,1504,3))
# djj_1[0,:] = image_1[:]
# djj_2[0,:] = image_2[:]
# index = ssim.compare_ssim(djj_1,djj_2,win_size = 13,multichannel = True,gaussian_weights=True)
# x = np.array([[1,1],[2,2]])
# y = np.var(image_1)
# z = np.mean(image_1)
# w = np.cov(image_1[:,:,0],image_2[:,:,0])

#########################################################################################
patch = 3
expand = 1
image_1 = tf.placeholder(dtype=tf.float32, shape=[1, 256, 256, 1], name='x_img')  # 输入的x域图像
image_2 = tf.placeholder(dtype=tf.float32, shape=[1, 256, 256, 1], name='y_img')  # 输入的y域图像
# img_x_1 =tf.Variable( tf.zeros((256+patch-1,256+patch-1)),name = 'img_x_1' )
# img_x_2 =tf.Variable( tf.zeros((256+patch-1,256+patch-1)),name = 'img_x_2' )
# img_x_3 =tf.Variable( tf.zeros((256+patch-1,256+patch-1)),name = 'img_x_3' )
# img_y_1 =tf.Variable( tf.zeros((256+patch-1,256+patch-1)),name = 'img_y_1' )
# img_y_2 =tf.Variable( tf.zeros((256+patch-1,256+patch-1)),name = 'img_y_2'  )
# img_y_3 =tf.Variable( tf.zeros((256+patch-1,256+patch-1)),name = 'img_y_3'  )
#
# img_x_1 = tf.assign(img_x_1[expand:-expand,expand:-expand], image_1[0,:,:,0], validate_shape=True, use_locking=None, name=None)
# img_x_2 = tf.assign(img_x_2[expand:-expand,expand:-expand], image_1[0,:,:,1], validate_shape=True, use_locking=None, name=None)
# img_x_3 = tf.assign(img_x_3[expand:-expand,expand:-expand], image_1[0,:,:,2], validate_shape=True, use_locking=None, name=None)
#
# img_y_1 = tf.assign(img_y_1[expand:-expand,expand:-expand], image_2[0,:,:,0], validate_shape=True, use_locking=None, name=None)
# img_y_2 = tf.assign(img_y_2[expand:-expand,expand:-expand], image_2[0,:,:,1], validate_shape=True, use_locking=None, name=None)
# img_y_3 = tf.assign(img_y_3[expand:-expand,expand:-expand], image_2[0,:,:,2], validate_shape=True, use_locking=None, name=None)
# with tf.device("/gpu:0"):
# ssim_loss_1 = tf.Variable( 0,name = 'ssim_loss_1' )
k = 1
gauss_filter = np.array([1, 4, 7, 4, 1, 4, 16, 26, 16, 4, 7, 26, 41, 26, 7, 4, 16, 26, 16, 4, 1, 4, 7, 4, 1]) / 273.0
gauss_filter.reshape((5, 5, 1, 1))  ##我先用的5*5的滤波器###################################
gauss_filter = tf.constant(value=gauss_filter, shape=[5, 5, 1, 1], dtype=tf.float32, name='gauss_filter')
# gauss_filter=np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]])/273.0
# gauss_filter = tf.constant(gauss_filter,name='gauss_filter',dtype=tf.float32)
image_1_u = tf.nn.conv2d(image_1, gauss_filter, [1, 1, 1, 1], padding='SAME')
image_1_u2 = tf.multiply(image_1_u, image_1_u)

image_2_u = tf.nn.conv2d(image_2, gauss_filter, [1, 1, 1, 1], padding='SAME')
image_2_u2 = tf.multiply(image_2_u, image_2_u)

image_u1_u2 = tf.multiply(image_2_u, image_1_u)

var_image_1 = tf.nn.conv2d(tf.multiply(image_1, image_1), gauss_filter, [1, 1, 1, 1], padding='SAME') - image_1_u2
var_image_2 = tf.nn.conv2d(tf.multiply(image_2, image_2), gauss_filter, [1, 1, 1, 1], padding='SAME') - image_2_u2
var_image_12 = tf.nn.conv2d(tf.multiply(image_1, image_2), gauss_filter, [1, 1, 1, 1], padding='SAME') - image_u1_u2
# ssim_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2));
c_1 = 0.02
c_2 = 0.03
ssim_map = tf.multiply(tf.divide((2 * image_u1_u2 + c_1), (image_1_u2 + image_2_u2 + c_1)),
                       tf.divide((2 * var_image_12 + c_2), (var_image_1 + var_image_2 + c_2)))
ssim_ch1 = 1 - tf.reduce_mean(ssim_map)
# for i in range(256):
#    for j in range(256):
#        u_x_1 = tf.reduce_mean(img_x_1[i:i+patch,j:j+patch])
#        u_y_1 = tf.reduce_mean(img_y_1[i:i+patch,j:j+patch])
#        v_x_1 = tf.reduce_mean((img_x_1[i:i+patch,j:j+patch]-u_x_1)**2)
#        v_y_1 = tf.reduce_mean((img_y_1[i:i+patch,j:j+patch]-u_y_1)**2)
#        c_v_1 = tf.reduce_mean( (img_y_1[i:i+patch,j:j+patch]-u_y_1)*(img_x_1[i:i+patch,j:j+patch]-u_x_1) )
#        print(k)
#        k+=1


##########################################################################################
# image_1 = tf.placeholder(dtype=tf.float32,shape=[256, 256,3],name='x_img') #输入的x域图像
# image_2 = tf.placeholder(dtype=tf.float32,shape=[256, 256,3],name='y_img') #输入的y域图像
################image sizes are n*row*col*channel
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 设定显存不超量使用
sess = tf.Session(config=config)  # 新建会话层
init = tf.global_variables_initializer()  # 参数初始化器
sess.run(init)  # 初始化所有可训练参数

patch = 3
a = np.ones((1, 256, 256, 1))
b = np.zeros((1, 256, 256, 1))
ha = sess.run(ssim_ch1, feed_dict={image_1: a, image_2: b})
