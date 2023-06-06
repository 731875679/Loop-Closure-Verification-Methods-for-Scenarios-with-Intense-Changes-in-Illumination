import os
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

from PyQt5.QtCore import QLibraryInfo
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)



def brute_force_matcher(des1, des2):
    """
    Brute force matcher to match ORB feature descriptors
    """
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    return matches

def brute_force_matcher2(des1, des2):
    """
    Brute force matcher to match ORB feature descriptors
    """
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    return matches

def compute_sift_keypoints1(img):
    """
    Reads image from filename and computes ORB keypoints
    Returns image, keypoints and descriptors. 
    """
    
    # create orb object
    sift = cv2.SIFT_create()
    
    kp, des = sift.detectAndCompute(img, None)
    
    return img,kp, des

def compute_orb_keypoints2(img):
    """
    Reads image from filename and computes ORB keypoints
    Returns image, keypoints and descriptors. 
    """
    # create orb object
    orb = cv2.ORB_create()
    
    # set parameters 
    orb.setScoreType(cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
    orb.setWTA_K(3)
    
    # detect keypoints
    kp = orb.detect(img,None)

    # for detected keypoints compute descriptors. 
    kp, des = orb.compute(img, kp)
    
    return img,kp, des


def compute_sift_keypoints3(img_rgb,img_dep):
    """
    Reads image from filename and computes ORB keypoints
    Returns image, keypoints and descriptors. 
    """


    # Initialize SIFT feature detector and descriptor
    sift1 = cv2.SIFT_create()
    sift2 = cv2.SIFT_create()
    sift1.setContrastThreshold(0.01)
    sift1.setEdgeThreshold(50) 
    sift2.setContrastThreshold(0.01)
    sift2.setEdgeThreshold(50)       
    # Detect keypoints and compute descriptors for RGB image
    keypoints_rgb, descriptors_rgb = sift1.detectAndCompute(img_rgb, None)

    # Resize depth map to match the dimensions of the RGB image
    resized_depth_map = cv2.resize(img_dep, (img_rgb.shape[1], img_rgb.shape[0]))

    # Convert depth map to float32 for calculations
    depth_map_float = resized_depth_map.astype(np.float32)

    # Normalize depth map values to the range [0, 1]
    normalized_depth = cv2.normalize(depth_map_float, None, 0, 1, cv2.NORM_MINMAX)

    # Convert normalized depth map to grayscale
    grayscale_depth = (normalized_depth * 255).astype(np.uint8)

    # Detect keypoints and compute descriptors for depth map
    keypoints_depth, descriptors_depth = sift2.detectAndCompute(grayscale_depth, None)

    # Combine keypoints and descriptors from RGB and depth images
    keypoints_combined = keypoints_rgb + keypoints_depth
    if len(keypoints_depth)==0:
        descriptors_combined = descriptors_rgb
    else:
        descriptors_combined = np.concatenate((descriptors_rgb, descriptors_depth), axis=0)

    return img_rgb,keypoints_combined, descriptors_combined

def compute_fundamental_matrix(filename1_rgb,filename1_dep, filename2_rgb,filename2_dep,mode=None):
    """
    Takes in filenames of two input images 
    Return Fundamental matrix computes 
    using 8 point algorithm
    """
    # read images
    img1_rgb = cv2.imread(filename1_rgb)
    img2_rgb = cv2.imread(filename2_rgb)
    # get grayscale images
    img1_rgb = cv2.cvtColor(img1_rgb, cv2.COLOR_BGR2GRAY)
    img2_rgb = cv2.cvtColor(img2_rgb, cv2.COLOR_BGR2GRAY)

    # compute ORB keypoints and descriptor for each image
    if mode == 'sift':
        img1, kp1, des1 = compute_sift_keypoints1(img1_rgb)
        img2, kp2, des2 = compute_sift_keypoints1(img2_rgb)
        matches = brute_force_matcher(des1, des2)
        thres = 500
    elif mode == 'orb':
        img1, kp1, des1 = compute_orb_keypoints2(img1_rgb)
        img2, kp2, des2 = compute_orb_keypoints2(img2_rgb)
        matches = brute_force_matcher2(des1, des2)
        thres = 30
    else:
        img1_dep = cv2.imread(filename1_dep)
        img2_dep = cv2.imread(filename2_dep)
        img1_dep = cv2.cvtColor(img1_dep, cv2.COLOR_BGR2GRAY)
        img2_dep = cv2.cvtColor(img2_dep, cv2.COLOR_BGR2GRAY)

        img1, kp1, des1 = compute_sift_keypoints3(img1_rgb, img1_dep)
        img2, kp2, des2 = compute_sift_keypoints3(img2_rgb, img2_dep)
        matches = brute_force_matcher(des1, des2)
        thres = 500

    # extract points
    pts1 = []
    pts2 = []
    good_matches = []
    for i,(m) in enumerate(matches):
        if m.distance < thres:
            #print(m.distance)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            good_matches.append(matches[i])

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    
    # Compute fundamental matrix
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC, ransacReprojThreshold=3, confidence=0.99)

    # We select only inlier points
    if mask is None:
      print(0, end="\t")
      return False
    else: 
      inlier_matches = [b for a, b in zip(mask, good_matches) if a]

    # print(len(inlier_matches), end="\t")

    return len(inlier_matches)
    # else:
    #     if len(inlier_matches) > 20:     # set this condition to what you need 
    #         return True
    #     else:
    #         return False    
    
def evaluate(gt_txt,mode=None):
    match_points = []
    labels = []
    
    fp = open(gt_txt, "r")
    for line in tqdm(fp):
        line_str = line.split(", ")
        query, reference, gt = line_str[0], line_str[1], int(line_str[2])
        
        query_rgb='./data_rgb/'+query
        reference_rgb='./data_rgb/'+reference
        
        if mode == 'sift':
            match_points.append(compute_fundamental_matrix(query_rgb, '' , reference_rgb, '' , 'sift'))
        elif mode == 'sift_dep':
            match_points.append(compute_fundamental_matrix(query_rgb,'./data_dep/'+query, reference_rgb , './data_dep/'+reference, 'sift_dep'))
        else:
            match_points.append(compute_fundamental_matrix(query_rgb, '' , reference_rgb, '' , 'orb'))

        labels.append(gt)
    return np.array(match_points), np.array(labels)


def main():
    # filename = "./RobotCar/robotcar_qAutumn_dbNight_easy_final.txt"

    TP = 0
    P = 0    # P = TP + FN
    FP = 0
    # with open(filename) as file:
    #   for item in file:
    #     terms = item.split(' ')
    #     terms[0] = terms[0].rstrip(",")
    #     terms[1] = terms[1].rstrip(",")

    #     file1_rgb = './EE5346_2023_project-main/'+terms[0]        
    #     file1_dep = './EE5346_2023_project-main/data_dep/'+terms[0]

    #     file2_rgb = './EE5346_2023_project-main/'+terms[1]
    #     file2_dep = './EE5346_2023_project-main/data_dep/'+terms[1]

    #     GT = terms[2].rstrip('\n') == '1'

    #     TST = compute_fundamental_matrix(file1_rgb, file1_dep, file2_rgb, file2_dep)

    #     print(TST, '\t', GT, end="\t")
    #     if(GT == True):
    #       P = P + 1
          
    #       if(TST == True):
    #         TP = TP + 1
    #       if(TP == 0):
    #         continue
    #       print("Recall = {0:.2f},  Precision = {1:.2f} ".format(TP/P,TP/(FP+TP)) )

    #     else:
    #       if(TST == True):
    #         FP = FP + 1
    #       print("Recall = {0:.2f},  Precision = {1:.2f} ".format(TP/P, TP/(FP+TP)) )

    ### draw PR diagram ###
    
    # datasets = [ "robotcar_qAutumn_dbNight_easy_final.txt", "robotcar_qAutumn_dbNight_diff_final.txt"]

    datasets = [ "robotcar_qAutumn_dbSunCloud_easy_final.txt", "robotcar_qAutumn_dbSunCloud_diff_final.txt"]

    for dataset in datasets:
        print("-------- Processing {} ----------".format(dataset.strip(".txt")))
        match_points1, labels = evaluate(dataset,'orb')
        scaled_scores1 = match_points1 / max(match_points1)
        precision1, recall1, _ = precision_recall_curve(labels, scaled_scores1)

        match_points2, labels = evaluate(dataset,'sift')
        scaled_scores2 = match_points2 / max(match_points2)
        precision2, recall2, _ = precision_recall_curve(labels, scaled_scores2)

        match_points3, labels = evaluate(dataset,'sift_dep')
        scaled_scores3 = match_points3 / max(match_points3)
        precision3, recall3, _ = precision_recall_curve(labels, scaled_scores3)

        max_precision1 = np.argmax(precision1)
        print('highest recall of ORB is {:.3f} '.format(recall1[max_precision1]))
        max_precision2 = np.argmax(precision2)
        print('highest recall of SIFT is {:.3f} '.format(recall2[max_precision2]))    
        max_precision3 = np.argmax(precision3)
        print('highest recall of SIFT + Depth is {:.3f} '.format(recall3[max_precision3]))      
        # precision = precision[:-1]
        # recall = recall[:-1]
        average_precision1 = average_precision_score(labels, scaled_scores1)
        average_precision2 = average_precision_score(labels, scaled_scores2)
        average_precision3 = average_precision_score(labels, scaled_scores3)

        plt.plot(recall1, precision1, 'g',label="{} (AP={:.3f})".format('ORB', average_precision1))
        plt.plot(recall2, precision2, 'b',label="{} (AP={:.3f})".format('SIFT', average_precision2))
        plt.plot(recall3, precision3, 'r',label="{} (AP={:.3f})".format('SIFT + Depth', average_precision3))

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.title("Precision-Recall Curves for {}".format(dataset.strip(".txt")))
        plt.savefig("pr_curve_{}_5.png".format(dataset.strip(".txt")))
        plt.close()


if __name__ == '__main__':
    main()