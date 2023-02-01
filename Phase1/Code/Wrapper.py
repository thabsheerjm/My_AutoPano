
from pickle import FALSE
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
from os import listdir
import copy



def ANMS(image, N_best):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # find corners using shi-tomasi
    corners = cv.goodFeaturesToTrack(gray_image, 5000, 0.01, 10)
    corners = np.squeeze(corners)
    total_corners = corners.shape[0]
    corner_score = cv.cornerHarris(gray_image, 8, 3, 0.1)

    r = np.full((total_corners, 3), np.inf)
    for i in range(total_corners):
        xi, yi = int(corners[i][0]), int(corners[i][1])
        for j in range(total_corners):
            xj, yj = int(corners[j][0]), int(corners[j][1])
            ED = (xj - xi)**2 + (yj - yi)**2
            if corner_score[yj][xj] > corner_score[yi][xi] and ED < r[i][2]:
                r[i][0] = xi
                r[i][1] = yi
                r[i][2] = ED

    r = r[r[:, 2].argsort()][::-1]

    k = np.ones((N_best, 2))
    count = 0
    i = 0
    while count < N_best:
        if (corners[i][0] > 21 and corners[i][0] < image.shape[1] - 21
                and corners[i][1] > 21 and corners[i][1] < image.shape[0] - 21):
            k[count][0] = corners[i][0]
            k[count][1] = corners[i][1]
            count += 1
        i += 1

    features = k
    return features


def featurevector(image,corners):
    image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

    # Apply Gaussian blur for all the corner points
    Num_corners = corners.shape[0]
    std_featurevector = {}  
    
    for i in range(Num_corners):
        row = int(corners[i][1])
        col = int(corners[i][0])
        #40X40 Patch around the center of corner
        patch_i = image[row-20:row+20,col-20:col+20]
        patch = cv.GaussianBlur(patch_i,(5,5),cv.BORDER_DEFAULT) 
        res_patch = cv.resize(patch,(8,8))
        vector = np.resize(res_patch,(64,1))  
        # Standardization of the vector: subtract the vector by its mean, and then divide the result by the vector's standard deviation.
        std_featurevector[(col,row)] = (vector-np.mean(vector))/np.std(vector)
   
    return std_featurevector



def FeatureMatching(features1,features2,threshold):
    correspondence_list = [] # tuple of keypoints of image 1 and 2 that are a match
    # get the coordinate values and norm value of the vector if they are under a threshold
    for i in features1:
        norm_list = []
        for j in features2:
            norm_list.append((j, np.linalg.norm(features1[i]-features2[j])))
        norm_list.sort(key = lambda X:X[1])
        if ((norm_list[0][1])/norm_list[1][1])<threshold:
            correspondence_list.append([i,norm_list[0][0]])
        #    # edge case : On next step check if the list has more than 4 matches for random sampling during RANSAC
    # correspondence_list = False
    return correspondence_list


def Drawmatches(image1, image2, loc1,loc2):
    # color image
    dim = (max(image1.shape[0], image2.shape[0]), image1.shape[1]+image2.shape[1], image1.shape[2]) # dimensions of new image that has bothe im1 and im2 joined horizontally
    match_image = np.zeros(dim,type(image1.flat[0]))
    # join the images horizontally
    match_image[0:image1.shape[0],0:image1.shape[1]] = image1
    match_image[0:image2.shape[0],image1.shape[1]:image1.shape[1]+image2.shape[1]] = image2
    # and draw lines between the matching pairs (x1,y1)  to (x2,y2)
    for i in  range(len(loc1)):
        x1 = loc1[i][0]
        y1 = loc1[i][1]
        x2 = loc2[i][0] + int(image1.shape[1]) # horizontal shift
        y2 = loc2[i][1]
        cv.line(match_image,(x1,y1),(x2,y2),(0,255,255),1)
    cv.imshow('image',match_image)
    cv.waitKey(0)
    cv.destroyAllWindows
    return None

def Homography(kp1, kp2, threshold):

    #Estimate Homography matrix between the two images using RANSAC.    
    
    flag = True
    maxinlier = 0
    n_points = kp1.shape[0]
    
    # Find homography for a random set of 4 points
    for _ in range(6000):
        index = np.random.randint(n_points, size=4)
        pts1 = kp1[index]
        pts2 = kp2[index]
        H, status = cv.findHomography(pts1, pts2)
        
        # Count the number of inliers
        inlier = 0
        matched_pts1 = []
        matched_pts2 = []
        source_pts = np.column_stack([kp1, np.ones((n_points, 1))])
        actual_pts = np.column_stack([kp2, np.ones((n_points, 1))])
        predicted_pts = np.matmul(H, source_pts.T)
        predicted_pts = predicted_pts / predicted_pts[2, :]
        distances = np.linalg.norm(actual_pts - predicted_pts.T, axis=1)
        inlier_indices = np.where(distances < threshold)[0]
        if inlier_indices.size > maxinlier:
            maxinlier = inlier_indices.size
            final_H = H
            final_pts1 = source_pts[inlier_indices]
            final_pts2 = actual_pts[inlier_indices]
    
    print("Max Inlier = ", maxinlier)
    if maxinlier <= 4:
        print("Total Inliers are less than 4")
        flag = False
    final_H, status = cv.findHomography(final_pts1[:, :2], final_pts2[:, :2])

    return final_H, final_pts1, final_pts2, flag


def main():
    images=[]
    folder_dir = "../Data/Train/Set1/"
    for im in os.listdir(folder_dir):
        if (im.endswith(".jpg")):
            images.append(im)   
    
    images.sort()
    warped_images = []
    size_of_image = (1080,630)  # 1080 630
    resize_image_size = 800
    Translation = (100,150)
    Nbest = 300
    
    # #Initially the first image is the reference image
    ref_img = cv.imread("%s%s" % (folder_dir, images[0]))
    ref_img = cv.resize(ref_img,(resize_image_size,resize_image_size), interpolation = cv.INTER_AREA)
    # Ref_img = copy.deepcopy(ref_img)

    #ref_img_gray = cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY)
    ref_corner =  ANMS(ref_img, Nbest)
    ref_Feature = featurevector(ref_img,ref_corner)
    ref_H = np.matmul(np.array([[1,0,Translation[0]],[0,1,Translation[1]],[0,0,1]]),np.identity(3))
    warped_images.append(cv.warpPerspective(ref_img, ref_H, size_of_image))

    for i in range(1,len(images)):
        
        img = cv.imread("%s%s" % (folder_dir, images[i]))
        img = cv.resize(img, (resize_image_size,resize_image_size), interpolation = cv.INTER_AREA)
        corners = ANMS(img,Nbest)
        features = featurevector(img,corners)
        z = FeatureMatching(ref_Feature,features,threshold =0.6)
        kp1 = []
        kp2 = []
        for i in z:
            kp1.append(i[0])  # Location of keypoints in image 1
            kp2.append(i[1])
        Img = img[:,:]
        Drawmatches(ref_img,img,kp1,kp2)
        H,matched_pts1,matched_pts2,flag = Homography(kp1,kp2,threshold = 1.5)
        if flag == False:
            print("Not able to calculate Homography")
            break
        #print(matched_pts2)
        
        Drawmatches(ref_img,img,matched_pts1, matched_pts2)
        ref_H = np.matmul(ref_H,np.linalg.inv(H))
        warped_images.append(cv.warpPerspective(Img,ref_H, size_of_image))
        ref_img = np.copy(img)

        ref_Feature = features

    out =  np.zeros((size_of_image[1],size_of_image[0],3),dtype=np.uint8)
    temp = np.array([0,0,0],dtype=np.uint8)
    out = np.copy(warped_images[0])
    for img in range(1,len(warped_images)):
        for i in range(size_of_image[1]):
            for j in range(size_of_image[0]):
                if (np.array_equal(warped_images[img][i][j],temp) == False):
                    out[i][j] = warped_images[img][i][j]

    cv.imshow('ouput image',out)
    cv.waitKey()
    cv.destroyAllWindows()



if __name__ == '__main__':
    main()