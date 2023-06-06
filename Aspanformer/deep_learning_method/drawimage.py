
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import kornia as K
import kornia.feature as KF
from src.ASpanFormer.aspanformer import ASpanFormer 
from src.config.default import get_cfg_defaults
from src.utils.misc import lower_config
import demo_utils 
import argparse
import cv2
import torch
import numpy as np

def load_torch_image(fname,long_dim):
    img = cv2.imread(fname)
    h,w=img.shape[0],img.shape[1]
    image=cv2.resize(img,(int(w*long_dim/max(h,w)),int(h*long_dim/max(h,w))))
    img = K.image_to_tensor(image, False).float() /255.
    img = K.color.bgr_to_rgb(img)
    return img

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='C:/Users/admin/Desktop/course/2023/Autonomous Navigation/project/ml-aspanformer-main/configs/aspan/outdoor/aspan_test.py',
  help='path for config file.')
parser.add_argument('--img0_path', type=str, default='./EE5346_2023_project-main/Autumn_mini_query/1418133749992569.jpg',
  help='path for image0.')
parser.add_argument('--img1_path', type=str, default='./EE5346_2023_project-main/Night_mini_ref/1418756787288766.jpg',
  help='path for image1.')
parser.add_argument('--weights_path', type=str, default='C:/Users/admin/Desktop/course/2023/Autonomous Navigation/project/ml-aspanformer-main/weights/outdoor.ckpt',
  help='path for model weights.')
parser.add_argument('--long_dim0', type=int, default=640,
  help='resize for longest dim of image0.')
parser.add_argument('--long_dim1', type=int, default=640,
  help='resize for longest dim of image1.')

args = parser.parse_args()


def main():

    config = get_cfg_defaults()
    config.merge_from_file(args.config_path)
    _config = lower_config(config)
    matcher1 = ASpanFormer(config=_config['aspan'])
    state_dict = torch.load(args.weights_path, map_location='cpu')['state_dict']
    matcher1.load_state_dict(state_dict,strict=False)
    matcher1.cuda(),matcher1.eval()

    matcher2 = KF.LoFTR(pretrained='outdoor')


    img0,img1=cv2.imread(args.img0_path),cv2.imread(args.img1_path)
    img0_g,img1_g=cv2.imread(args.img0_path,0),cv2.imread(args.img1_path,0)
  
    img0,img1=demo_utils.resize(img0,args.long_dim0),demo_utils.resize(img1,args.long_dim1)
    img0_g,img1_g=demo_utils.resize(img0_g,args.long_dim0),demo_utils.resize(img1_g,args.long_dim1)
    data={'image0':torch.from_numpy(img0_g/255.)[None,None].cuda().float(),
            'image1':torch.from_numpy(img1_g/255.)[None,None].cuda().float()} 
        
    ### Aspanformer ###
    with torch.no_grad():   
        matcher1(data,online_resize=True)
        corr0,corr1=data['mkpts0_f'].cpu().numpy(),data['mkpts1_f'].cpu().numpy()
    mkpts0_aspan = corr0
    mkpts1_aspan = corr1

    ### LoFtr ###
    img2 = load_torch_image(args.img0_path,args.long_dim0)
    img3 = load_torch_image(args.img1_path,args.long_dim1)
    
    input_dict = {"image0": K.color.rgb_to_grayscale(img2), # LofTR 只在灰度图上作用
                    "image1": K.color.rgb_to_grayscale(img3)}
    with torch.inference_mode():
        corr2 = matcher2(input_dict)

    mkpts0_loftr = corr2['keypoints0'].cpu().numpy()
    mkpts1_loftr = corr2['keypoints1'].cpu().numpy()


    ### ensemble ###
    mkpts0_all = []
    mkpts1_all = []

    if len(mkpts0_loftr) > 0:
        mkpts0_all.append(mkpts0_loftr)
        mkpts1_all.append(mkpts1_loftr)
        
    if len(mkpts0_aspan) > 0:
        mkpts0_all.append(mkpts0_aspan)
        mkpts1_all.append(mkpts1_aspan)
        
    mkpts0_all = np.concatenate(mkpts0_all, axis=0)
    mkpts1_all = np.concatenate(mkpts1_all, axis=0) 

    F_hat,mask_F=cv2.findFundamentalMat(mkpts0_loftr,mkpts1_loftr,method=cv2.FM_RANSAC,ransacReprojThreshold=1)
    # F_hat,mask_F=cv2.findFundamentalMat(corr0,corr1,method=cv2.FM_RANSAC,ransacReprojThreshold=1)
    # F_hat,mask_F=cv2.findFundamentalMat(mkpts0_all,mkpts1_all,method=cv2.FM_RANSAC,ransacReprojThreshold=1)
    if mask_F is not None:
      mask_F=mask_F[:,0].astype(bool) 
    else:
      mask_F=np.zeros_like(corr0[:,0]).astype(bool)

    #visualize match
    # display=demo_utils.draw_match(img0,img1,mkpts0_loftr,mkpts1_loftr)
    # display_ransac=demo_utils.draw_match(img0,img1,mkpts0_loftr[mask_F],mkpts1_loftr[mask_F])

    print(len(mkpts0_loftr[mask_F]))
    # display=demo_utils.draw_match(img0,img1,corr0,corr1)
    # display_ransac=demo_utils.draw_match(img0,img1,corr0[mask_F],corr1[mask_F])
    # display=demo_utils.draw_match(img0,img1,mkpts0_all,mkpts1_all)
    # display_ransac=demo_utils.draw_match(img0,img1,mkpts0_all[mask_F],mkpts1_all[mask_F])
    # cv2.imwrite('match3.png',display)
    # cv2.imwrite('match_ransac3.png',display_ransac)

if __name__=='__main__':
    main()
