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



def compute_orb_keypoints(filename):
    """
    Reads image from filename and computes ORB keypoints
    Returns image, keypoints and descriptors. 
    """
    # load image
    img = cv2.imread(filename)
    
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

    
def brute_force_matcher(des1, des2):
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

def compute_fundamental_matrix(filename1, filename2, mode = None):
    """
    Takes in filenames of two input images 
    Return Fundamental matrix computes 
    using 8 point algorithm
    """
    # compute ORB keypoints and descriptor for each image
    img1, kp1, des1 = compute_orb_keypoints(filename1)
    img2, kp2, des2 = compute_orb_keypoints(filename2)
    
    # compute keypoint matches using descriptor
    matches = brute_force_matcher(des1, des2)
    
    # extract points
    pts1 = []
    pts2 = []
    good_matches = []
    for i,(m) in enumerate(matches):
        if m.distance < 20:
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

    print(len(inlier_matches), end="\t")

    if mode == 'draw':
      return len(inlier_matches)
    else:
        final_img = cv2.drawMatches(img1, kp1, img2, kp2, matches,None)
        # cv2.imshow("BF Matches", final_img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        final_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches,None)
        # cv2.imshow("Good Matches", final_img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        final_img = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches,None)
        # cv2.imshow("Inlier Matches", final_img)
        # if (cv2.waitKey(25) & 0xFF) == 27:
        #     cv2.destroyAllWindows()
        #     sys.exit() 
        # else:
        #     cv2.waitKey()
        #     cv2.destroyAllWindows()
        if len(inlier_matches) > 8:     # set this condition to what you need 
            return True
        else:
            return False    

def evaluate(gt_txt):
    match_points = []
    labels = []
    fp = open(gt_txt, "r")
    for line in tqdm(fp):
        line_str = line.split(", ")
        query, reference, gt = line_str[0], line_str[1], int(line_str[2])
        match_points.append(compute_fundamental_matrix(query, reference, 'draw'))
        labels.append(gt)
    return np.array(match_points), np.array(labels)

def main():
    filename = "./EE5346_2023_project-main/robotcar_qAutumn_dbNight_diff_final.txt"
    # filename = "./RobotCar/robotcar_qAutumn_dbNight_easy_final.txt"

    TP = 0
    P = 0    # P = TP + FN
    FP = 0
    with open(filename) as file:
      for item in file:
        terms = item.split(' ')
        terms[0] = terms[0].rstrip(",")
        terms[1] = terms[1].rstrip(",")
        file1 = './EE5346_2023_project-main/'+terms[0]
        file2 = './EE5346_2023_project-main/'+terms[1]
        GT = terms[2].rstrip('\n') == '1'
        TST = compute_fundamental_matrix(file1, file2)

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

import cv2
import numpy as np

def extract_feature_points(rgb_image, depth_map):
    # Convert RGB image to grayscale
    grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT feature detector and descriptor
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
    descriptors_combined = np.concatenate((descriptors_rgb, descriptors_depth), axis=0)

    return keypoints_combined, descriptors_combined

# Load RGB images and depth maps
rgb_image1 = cv2.imread('rgb_image1.jpg')
depth_map1 = cv2.imread('depth_map1.jpg', cv2.IMREAD_GRAYSCALE)
rgb_image2 = cv2.imread('rgb_image2.jpg')
depth_map2 = cv2.imread('depth_map2.jpg', cv2.IMREAD_GRAYSCALE)

# Extract feature points using RGB and depth information
feature_points1, descriptors1 = extract_feature_points(rgb_image1, depth_map1)
feature_points2, descriptors2 = extract_feature_points(rgb_image2, depth_map2)

# Create a brute-force matcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match the descriptors
matches = bf.match(descriptors1, descriptors2)

# Sort the matches based on distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw the top 10 matches
matching_result = cv2.drawMatches(rgb_image1, feature_points1, rgb_image2, feature_points2, matches[:10], None, flags=2)

# Display the matching result
cv2.imshow('Matching Result', matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
