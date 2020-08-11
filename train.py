#!/usr/bin/env python3

if __name___=='__main__':
  import singan
  import sys
  import os
  import cv2
  
  if len(sys.argv) < 3:
    print(f'Usage: {sys.argv[0]} [image_file] [save_dir]')
    quit()
  image_file = sys.argv[1]
  save_dir = sys.argv[2]
    
  gan = singan.SinGAN()
  image = cv2.imread(image_file)[:,:,::-1]
  
  if os.path.exists(save_dir):
    gan.restore(save_dir)
  else:
    os.mkdir(save_dir)
  
  for i in range(len(gan.gans)):
    gan.train_stage(i, image)
  gan.save(save_dir)
