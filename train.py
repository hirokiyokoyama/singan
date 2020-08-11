#!/usr/bin/env python3

if __name__=='__main__':
  import singan
  import sys
  import os
  import cv2
  import tensorflow as tf
  
  if len(sys.argv) < 3:
    print(f'Usage: {sys.argv[0]} [image_file] [save_dir]')
    quit()
  image_file = sys.argv[1]
  save_dir = sys.argv[2]
    
  gan = singan.SinGAN()
  image = cv2.imread(image_file)[:,:,::-1]
  h = image.shape[0]
  w = image.shape[1]
  size = min(h, w)
  top = (h-size)//2
  left = (w-size)//2
  image = image[top:top+size,left:left+size,:]
  image = tf.constant(image)[tf.newaxis]
  image = tf.image.resize(image, gan.default_output_size[-1])
  
  if os.path.exists(save_dir):
    gan.restore(save_dir)
  else:
    os.mkdir(save_dir)
  
  for i in range(len(gan.gans)):
    gan.train_stage(i, image)
  gan.save(save_dir)
