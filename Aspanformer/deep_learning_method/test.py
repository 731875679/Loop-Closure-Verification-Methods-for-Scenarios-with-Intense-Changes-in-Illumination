import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
import glob
import kornia as K
import kornia.feature as KF
from collections import deque
from kornia_moons.feature import *
from scipy.spatial import Delaunay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from PIL import Image
import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
from src.ASpanFormer.aspanformer import ASpanFormer 
from src.config.default import get_cfg_defaults
from src.utils.misc import lower_config
import torch
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='C:/Users/admin/Desktop/course/2023/Autonomous Navigation/project/ml-aspanformer-main/configs/aspan/outdoor/aspan_test.py',
  help='path for config file.')
parser.add_argument('--weights_path', type=str, default='C:/Users/admin/Desktop/course/2023/Autonomous Navigation/project/ml-aspanformer-main/weights/outdoor.ckpt',
  help='path for model weights.')
parser.add_argument('--long_dim0', type=int, default=640,
  help='resize for longest dim of image0.')
parser.add_argument('--long_dim1', type=int, default=640,
  help='resize for longest dim of image1.')

args = parser.parse_args()

def resize(image,long_dim):
    h,w=image.shape[0],image.shape[1]
    image=cv2.resize(image,(int(w*long_dim/max(h,w)),int(h*long_dim/max(h,w))))
    return image

def load_torch_image(fname,long_dim):
    img = cv2.imread(fname)
    h,w=img.shape[0],img.shape[1]
    image=cv2.resize(img,(int(w*long_dim/max(h,w)),int(h*long_dim/max(h,w))))
    img = K.image_to_tensor(image, False).float() /255.
    img = K.color.bgr_to_rgb(img)
    return img

def compute_fundamental_matrix(corr1, corr2, mode=None):

    # Compute fundamental matrix
    # F, mask = cv2.findFundamentalMat(corr1,corr2,cv2.FM_RANSAC, ransacReprojThreshold=0.15, confidence=0.99)
    F, mask = cv2.findFundamentalMat(corr1, corr2, cv2.USAC_MAGSAC, 1, 0.9999, 20000)

    # We select only inlier points
    if mask is None:
      print(0, end="\t")
      return False
    else: 
      inlier_matches = len(corr2[mask])

    # print(inlier_matches, end="\t")

    if mode == 'all':
      if inlier_matches > 800:     # combined
          return True, inlier_matches
      else:
          return False, inlier_matches 
    elif mode == 'aspan':
      if inlier_matches > 700:     # select condition
          return True, inlier_matches
      else:
          return False, inlier_matches 
    else:
      if inlier_matches > 300:     # select condition
          return True, inlier_matches
      else:
          return False, inlier_matches


def compute_similarity(corr1, corr2):
    tri1 = Delaunay(corr1)
    tri2 = Delaunay(corr2)

    # method 1 #
    triangles1 = tri1.simplices
    triangles2 = tri2.simplices
    # Count the number of shared triangles
    shared_triangles = len(set(map(tuple, triangles1)) & set(map(tuple, triangles2)))
    # Calculate the similarity score
    similarity = shared_triangles / max(len(triangles1), len(triangles2))
  

    # method 2 #
        
    # Perform breadth-first search on both triangulations
    # matched_edges = 0
    
    # for start_index in range(len(tri1.points)):
    #     visited = set()
    #     queue = deque([start_index])
        
    #     while queue:
    #         current_index = queue.popleft()
    #         visited.add(current_index)
            
    #         for neighbor in tri1.neighbors[current_index]:
    #             if neighbor in tri2.neighbors[current_index] and neighbor not in visited:
    #                 matched_edges += 1
    #                 queue.append(neighbor)
    
    # # Calculate similarity based on matched edges count
    # similarity = matched_edges / max(len(tri1.simplices), len(tri2.simplices))
    
    print(similarity, end="\t")
    if similarity > 0.18:     # set this condition to what you need 
        return True, similarity
    else:
        return False, similarity

def main():
    filename = "C:/Users/admin/Desktop/course/2023/Autonomous Navigation/project/EE5346_2023_project-main/robotcar_qAutumn_dbSuncloud_val_final.txt"
    # filename = "./RobotCar/robotcar_qAutumn_dbNight_easy_final.txt"

    TP = 0
    P = 0    # P = TP + FN
    FP = 0
    config = get_cfg_defaults()
    config.merge_from_file(args.config_path)
    _config = lower_config(config)
    matcher1 = ASpanFormer(config=_config['aspan'])
    state_dict = torch.load(args.weights_path, map_location='cpu')['state_dict']
    matcher1.load_state_dict(state_dict,strict=False)
    matcher1.cuda(),matcher1.eval()

    matcher2 = KF.LoFTR(pretrained='outdoor')

    match_points1 = []
    match_points2 = []
    match_points3 = []
    labels = []
    count = 0
    with open(filename) as file:
      with open('dbSuncloud_result.txt','w') as f2:
        for item in file:
          count += 1
          print(count)
          terms = item.split(' ')
          terms[0] = terms[0].rstrip(",")
          terms[1] = terms[1].rstrip(",")
          if '\n' in terms[0]:
            terms[0] =terms[0].replace('\n','')
          if '\n' in terms[1]:
            terms[1] =terms[1].replace('\n','')            
          file1 = './EE5346_2023_project-main/'+terms[0]
          file2 = './EE5346_2023_project-main/'+terms[1]



          img0_g = cv2.imread(file1,0)
          img1_g = cv2.imread(file2,0)


          img0_g,img1_g=resize(img0_g,args.long_dim0),resize(img1_g,args.long_dim1)
          data={'image0':torch.from_numpy(img0_g/255.)[None,None].cuda().float(),
              'image1':torch.from_numpy(img1_g/255.)[None,None].cuda().float()} 
          
          ### Aspanformer ###
          with torch.no_grad():   
              matcher1(data,online_resize=True)
              corr0,corr1=data['mkpts0_f'].cpu().numpy(),data['mkpts1_f'].cpu().numpy()
          mkpts0_aspan = corr0
          mkpts1_aspan = corr1


          ### LoFtr ###
          # img2 = load_torch_image(file1,args.long_dim0)
          # img3 = load_torch_image(file2,args.long_dim1)
      
          # input_dict = {"image0": K.color.rgb_to_grayscale(img2), # LofTR 只在灰度图上作用
          #             "image1": K.color.rgb_to_grayscale(img3)}
          # with torch.inference_mode():
          #     corr2 = matcher2(input_dict)

          # mkpts0_loftr = corr2['keypoints0'].cpu().numpy()
          # mkpts1_loftr = corr2['keypoints1'].cpu().numpy()


          # ### ensemble ###
          # mkpts0_all = []
          # mkpts1_all = []

          # if len(mkpts0_loftr) > 0:
          #     mkpts0_all.append(mkpts0_loftr)
          #     mkpts1_all.append(mkpts1_loftr)
          
          # if len(mkpts0_aspan) > 0:
          #     mkpts0_all.append(mkpts0_aspan)
          #     mkpts1_all.append(mkpts1_aspan)
          
          # mkpts0_all = np.concatenate(mkpts0_all, axis=0)
          # mkpts1_all = np.concatenate(mkpts1_all, axis=0) 

          # F_hat,mask_F=cv2.findFundamentalMat(corr0,corr1,method=cv2.FM_RANSAC,ransacReprojThreshold=1)

          ### method-fun_matrix ###
          # GT = terms[2].rstrip('\n') == '1'
          # TST,num1 = compute_fundamental_matrix(mkpts0_loftr, mkpts1_loftr,'loftr')
          TST,num2 = compute_fundamental_matrix(mkpts0_aspan, mkpts1_aspan,'aspan')
          # TST,num3 = compute_fundamental_matrix(mkpts0_all, mkpts1_all,'all')
          
          if(TST == True):
            f2.write("{}, {}, 1 ".format(terms[0] ,terms[1]))
            f2.write('\n')
          else:
            f2.write("{}, {}, 0".format(terms[0] ,terms[1]))
            f2.write('\n')
          ### method-graph_model ###
          # TST , num = compute_similarity(corr0, corr1)

          # match_points1.append(num1)
          # match_points2.append(num2)
          # match_points3.append(num3)
          # labels.append(GT)
          
          ## Compute precision and recall ###
          # if(GT == True):
          #   P = P + 1
          #   if(TST == True):
          #     TP = TP + 1
          #   if TP == 0:
          #     continue
          #   print("Recall = {0:.2f},  Precision = {1:.2f} ".format(TP/P,TP/(FP+TP)) )
  
          # else:
          #   if(TST == True):
          #     FP = FP + 1
          #   print("Recall = {0:.2f},  Precision = {1:.2f} ".format(TP/P,TP/(FP+TP)) )
    

    ### draw PR curves ###
    # match_points1 = np.array(match_points1)
    # match_points2 = np.array(match_points2)
    # match_points3 = np.array(match_points3)

    # labels = np.array(labels)
    # scaled_scores1 = match_points1 / max(match_points1)
    # scaled_scores2 = match_points2 / max(match_points2)
    # scaled_scores3 = match_points3 / max(match_points3)

    # precision1, recall1, _ = precision_recall_curve(labels, scaled_scores1)
    # precision2, recall2, _ = precision_recall_curve(labels, scaled_scores2)
    # precision3, recall3, _ = precision_recall_curve(labels, scaled_scores3)
    # # precision = precision[:-1]
    # # recall = recall[:-1]

    # # count highest recall #
    # max_precision1 = np.argmax(precision1)
    # print('highest recall of LoFTR is {:.3f} '.format(recall1[max_precision1]))
    # max_precision2 = np.argmax(precision2)
    # print('highest recall of Aspanformer is {:.3f} '.format(recall2[max_precision2])) 
    # max_precision3 = np.argmax(precision3)
    # print('highest recall of Aspanformer + LoFTR is {:.3f} '.format(recall3[max_precision3])) 

    # average_precision1 = average_precision_score(labels, scaled_scores1)
    # average_precision2 = average_precision_score(labels, scaled_scores2)

    # plt.plot(recall1, precision1,'g', label="{} (AP={:.3f})".format('LoFTR', average_precision2))
    # plt.plot(recall2, precision2,'b', label="{} (AP={:.3f})".format('Aspanformer', average_precision1))
    # plt.plot(recall3, precision3,'r', label="{} (AP={:.3f})".format('Aspanformer + LoFTR', average_precision2))
    
    # plt.xlabel("Recall")
    # plt.ylabel("Precision")
    # plt.legend()
    # plt.title("Precision-Recall Curves for robotcar_qAutumn_dbSunCloud_diff")
    # plt.savefig("pr_curve_{}.png".format('robotcar_qAutumn_dbSunCloud_diff_3model2'))
    # plt.close()


if __name__ == '__main__':
    main()