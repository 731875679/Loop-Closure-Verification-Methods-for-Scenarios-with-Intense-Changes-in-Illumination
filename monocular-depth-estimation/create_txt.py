# save all file name in a txt file

import os

dirname='/home/wang/Desktop/project/Monocular-Depth-Estimation-Toolbox/data/kitti/input/2011_09_26/2011_09_26_drive_0002_sync/image_02/data'
##只读取文件夹下一级文件名和子文件夹，并不会列举子文件夹里文件
names = os.listdir(dirname)

# for name in names:
#     path = os.path.join(dirname, name)  ##很有必要，不然结果会不对
#     if os.path.isdir(path): #文件夹
#         print(name, " is dir")
#     if os.path.isfile(path): #文件
#         print(name, " is file")

# save the path in the txt file
with open('/home/wang/Desktop/project/Monocular-Depth-Estimation-Toolbox/data/kitti/kitti_eigen_test.txt','w') as f:
    # 递归获得文件夹和子文件夹下所有文件名
    for root,dirs,files in os.walk(dirname):
        for file in files:
            path = os.path.join(root,file)
            # delimate the path by /, and get the last one
            path = path.split('/')
            # add /
            path = os.path.join(path[-5],path[-4],path[-3],path[-2],path[-1])
            f.write(path+' '+path+' 721.5377'+'\n')
f.close()
            

