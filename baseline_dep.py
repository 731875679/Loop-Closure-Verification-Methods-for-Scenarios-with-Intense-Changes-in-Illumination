import sys
import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
import glob
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from tqdm import tqdm
# With jupyter notebook uncomment below line 
# %matplotlib inline 
# This plots figures inside the notebook

import threading



def compute_sift_keypoints2(filename_rgb,filename_dep):
    """
    Reads image from filename and computes ORB keypoints
    Returns image, keypoints and descriptors. 
    """
    # load image
    img_rgb = cv2.imread(filename_rgb)
    depth_map = cv2.imread(filename_dep, cv2.IMREAD_GRAYSCALE)

    # Convert RGB image to grayscale
    grayscale_image = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    
    # Initialize ORB feature detector and descriptor
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for RGB image
    keypoints_rgb, descriptors_rgb = sift.detectAndCompute(grayscale_image, None)
    
    # Convert depth map to float32 for calculations
    depth_map = depth_map.astype(np.float32)

    # Normalize depth map values to the range [0, 255]
    normalized_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)

    # Convert normalized depth map to grayscale
    grayscale_depth = normalized_depth.astype(np.uint8)

    # Detect keypoints and compute descriptors for depth map
    keypoints_depth, descriptors_depth = sift.detectAndCompute(grayscale_depth, None)

    # Combine keypoints and descriptors from RGB and depth images
    keypoints_combined = keypoints_rgb + keypoints_depth

    if len(keypoints_depth)==0:
        descriptors_combined = descriptors_rgb
    else:
        descriptors_combined = np.concatenate((descriptors_rgb, descriptors_depth), axis=0)


    return img_rgb,keypoints_combined, descriptors_combined

def compute_sift_keypoints1(filename_rgb):
    """
    Reads image from filename and computes ORB keypoints
    Returns image, keypoints and descriptors. 
    """
    # load image
    img_rgb = cv2.imread(filename_rgb)

    #  Convert RGB image to grayscale
    grayscale_image = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    
    # Initialize ORB feature detector and descriptor
    sift = cv2.ORB_create()

    # Detect keypoints and compute descriptors for RGB image
    keypoints_rgb, descriptors_rgb = sift.detectAndCompute(grayscale_image, None)

    return img_rgb,keypoints_rgb, descriptors_rgb

def brute_force_matcher(des1, des2):
    """
    Brute force matcher to match ORB feature descriptors
    """
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    return matches

def select_matches(matches,kp1,kp2):
        # extract points
    pts1 = []
    pts2 = []
    good_matches = []
    for i,(m) in enumerate(matches):
        if m.distance < 1500:
            #print(m.distance)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            good_matches.append(matches[i])

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    
    # Compute fundamental matrix
    # F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC, ransacReprojThreshold=1, confidence=0.99)
    # F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.USAC_MAGSAC, 0.15, 0.9999, 20000)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC, ransacReprojThreshold=1, confidence=0.99)

    # We select only inlier points
    if mask is None:
      print(0, end="\t")
      return False
    else: 
      inlier_matches = [b for a, b in zip(mask, good_matches) if a]
    return inlier_matches

def show_matches(img1,kp1,des1,img2,kp2,des2,Mode=None):

    # compute keypoint matches using descriptor
    matches = brute_force_matcher(des1, des2)
    inlier_matches = select_matches(matches,kp1,kp2)
    print(len(inlier_matches), end="\t")

    if Mode == 'draw':
      return len(inlier_matches)
    else:
        # use green color to draw matches
        final_img = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches,None,matchColor=(0,255,0))

    if len(inlier_matches) > 8:     # set this condition to what you need 
        return final_img
    else:
        return False   
      
def compute_fundamental_matrix(file1_rgb,file2_rgb,Mode=None):
    """
    Takes in filenames of two input images 
    Return Fundamental matrix computes 
    using 8 point algorithm
    """
    img1_base,kp1_base, des1_base = compute_sift_keypoints1(file1_rgb)
    img2_base,kp2_base, des2_base = compute_sift_keypoints1(file2_rgb)

    # compute ORB keypoints and descriptor for each image

    # print the number of keypoints detected in the training image
    print("Number of Keypoints Detected In The base Image: ", len(kp1_base))
          
    # multi threading, show 2 matches at the same time
    img_base=show_matches(img1_base,kp1_base,des1_base,img2_base,kp2_base,des2_base,Mode)

    cv2.imshow("Inlier Matches Base", img_base)

    if (cv2.waitKey(25) & 0xFF) == 27:
        cv2.destroyAllWindows()
        sys.exit() 
    else:
        cv2.waitKey()
        cv2.destroyAllWindows()


# def evaluate(gt_txt):
#     match_points = []
#     labels = []
#     fp = open(gt_txt, "r")
#     for line in tqdm(fp):
#         line_str = line.split(", ")
#         query, reference, gt = line_str[0], line_str[1], int(line_str[2])
#         match_points.append(compute_fundamental_matrix(query, reference, 'draw'))
#         labels.append(gt)
        
#     return np.array(match_points), np.array(labels)

def main():
    filename = "/home/wang/Desktop/project/Monocular-Depth-Estimation-Toolbox/robotcar_qAutumn_dbNight_diff_final.txt"
    # filename = "./RobotCar/robotcar_qAutumn_dbNight_easy_final.txt"

    TP = 0
    P = 0    # P = TP + FN
    FP = 0
    with open(filename) as file:
      for item in file:
        terms = item.split(' ')
        terms[0] = terms[0].rstrip(",")
        terms[1] = terms[1].rstrip(",")

        file1_rgb = '/home/wang/Desktop/project/Monocular-Depth-Estimation-Toolbox/'+terms[0]

        file2_rgb = '/home/wang/Desktop/project/Monocular-Depth-Estimation-Toolbox/'+terms[1]

        GT = terms[2].rstrip('\n') == '1'

        TST = compute_fundamental_matrix(file1_rgb,file2_rgb)

        print(TST, '\t', GT, end="\t")
        if(GT == True):
          P = P + 1
          
          if(TST == True):
            TP = TP + 1
          if(TP == 0):
            continue
          print("Recall = {0:.2f},  Precision = {1:.2f} ".format(TP/P,TP/(FP+TP)) )

        else:
          if(TST == True):
            FP = FP + 1
          print("Recall = {0:.2f},  Precision = {1:.2f} ".format(TP/P, TP/(FP+TP)) )

    ### draw PR diagram ###
#     datasets = ["Kudamm_easy_final.txt", "Kudamm_diff_final.txt", "robotcar_qAutumn_dbNight_easy_final.txt", "robotcar_qAutumn_dbNight_diff_final.txt", "robotcar_qAutumn_dbSunCloud_easy_final.txt", "robotcar_qAutumn_dbSunCloud_diff_final.txt"]
# # 

#     for dataset in datasets:
#         print("-------- Processing {} ----------".format(dataset.strip(".txt")))
#         match_points, labels = evaluate(dataset)
#         scaled_scores = match_points / max(match_points)
#         precision, recall, _ = precision_recall_curve(labels, scaled_scores)
#         # precision = precision[:-1]
#         # recall = recall[:-1]
#         average_precision = average_precision_score(labels, scaled_scores)
        # plt.plot(recall, precision, label="{} (AP={:.3f})".format(dataset.strip(".txt"), average_precision))
        # plt.xlabel("Recall")
        # plt.ylabel("Precision")
        # plt.legend()
        # plt.title("Precision-Recall Curves for ORB baseline")
        # plt.savefig("pr_curve_{}.png".format(dataset.strip(".txt")))
        # plt.close()


if __name__ == '__main__':
    main()
