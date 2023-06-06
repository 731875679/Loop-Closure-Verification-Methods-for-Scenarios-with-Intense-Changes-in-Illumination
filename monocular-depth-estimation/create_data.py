

# resize the image under this directory to 1216*352

import os
import cv2
import numpy as np
import glob

def get_Listfiles(path):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for file in files:
            # include path
            Filelist.append(os.path.join(home, file))
            #Filelist.append(file)

    return Filelist

def resize_img(img_path):
    target_path = '/home/wang/Desktop/project/Monocular-Depth-Estimation-Toolbox/data/kitti/input/2011_09_26/2011_09_26_drive_0002_sync/image_02/data/'
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    # get the file name
    img_name = img_path.split('/')[-1]
    img = cv2.imread(img_path)
    img = cv2.resize(img, (1216, 352), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(target_path+img_name, img)

if __name__ == '__main__':
    #img_dir = '/home/wang/Desktop/project/dataset'
    img_dir='/home/wang/Desktop/project/Monocular-Depth-Estimation-Toolbox/data_rgb/Night_mini_ref'
    # read each file under this directory        
    img_list = get_Listfiles(img_dir)
    for img_path in img_list:
        resize_img(img_path)
