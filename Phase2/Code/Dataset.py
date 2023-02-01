import numpy as np
import cv2 as cv
import copy
import random
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def Generate_Train_Dataset(model_type = 'any'):
    
    Num_Trainset = 5000
    ImBatch =[]
    LabelBatch =[]
    Images = []
    Corners =[]
    Patches = []
    cropb = []
    for i in tqdm(range(1,Num_Trainset+1)):
        # Read the images
        img_name = str(i)+".jpg"
        img1 = cv.imread(img_name)
        Im = copy.deepcopy(img1)
        p = 32 #perturbation factor

        ''' Image Preprocessing  '''
        if len(Im.shape)==3:
            Im = cv.cvtColor(Im,cv.COLOR_BGR2GRAY)  # convert color images to gray
        Im = cv.resize(Im, (320, 240)) # RESIZE THE GRAY IMAGE
        Im = (Im-np.mean(Im))/255 # normalize the image

        '''Get a random Patch'''
        patch_width = 128
        patch_height =128

        # Random Origin points for the patch
        x = random.randrange(p, Im.shape[1]-p-patch_width-20)
        y = random.randrange(p, Im.shape[0]-p-patch_height-20)

        patch = np.array([[x,y],[x+patch_width,y],[x+patch_width,y+patch_height],[x,y+patch_height]], dtype=np.float32)
        perturbed_patch = []

        # perform a random perturbation on the patch in the range [-p,p]
        for corner in patch:
            perturbed_patch.append([corner[0]+random.randrange(-p,p),corner[1]+random.randrange(-p,p)])
        perturbed_patch = np.array(perturbed_patch,dtype =np.float32)

        #Estimate Homography
        H_AB = cv.getPerspectiveTransform(patch,perturbed_patch)  # Homography matrix
        H_BA = np.linalg.inv(H_AB)   # And its Inverse

        # warp the image
        warpped_image = cv.warpPerspective(Im, H_BA, (Im.shape[1],Im.shape[0]))

        #extract the patches from warpped image
        cropped_A = Im[y:y+patch_height, x:x+patch_width]
        cropped_B = warpped_image[y:y+patch_height, x:x+patch_width]
        cropped_A = cropped_A.astype(np.double)
       
        # initializa modelparameters
        # generate dataset
        data = np.stack((cropped_A,cropped_B))
        

        h4pt = np.subtract(perturbed_patch,patch)
        h4pt = h4pt.reshape(8,-1)
        # restricting to the range (-1,1)
        h4pt = h4pt/32
        # Append all images and mask
        ImBatch.append(data)
        LabelBatch.append(h4pt)
        Im = cv.resize(Im,(128,128))
        Images.append(np.float32(Im))
        Patches.append(data)
        Corners.append(np.float32(patch))
        cropb.append(np.float32(cropped_B.reshape(128,128,1)))
        # ImBatch training input 2images set
        # Label Batch h4pt
        #Patches are CA (corners of Patch A)
    return ImBatch, LabelBatch, Images, Corners, cropb


def Generate_Test_Dataset(model_type = 'any'):
    
    Num_Testset = 1000
    ImBatch =[]
    LabelBatch =[]
    Images = []
    Corners =[]
    Patches = []
    cropb = []
    for i in tqdm(range(1,Num_Testset+1)):
        # Read the images
        img_name = str(i)+".jpg"
        img1 = cv.imread(img_name)
        Im = copy.deepcopy(img1)
        p = 32 #perturbation factor

        ''' Image Preprocessing  '''
        if len(Im.shape)==3:
            Im = cv.cvtColor(Im,cv.COLOR_BGR2GRAY)  # convert color images to gray
        Im = cv.resize(Im, (320, 240)) # RESIZE THE GRAY IMAGE
        Im = (Im-np.mean(Im))/255 # normalize the image

        '''Get a random Patch'''
        patch_width = 128
        patch_height =128

        # Random Origin points for the patch
        x = random.randrange(p, Im.shape[1]-p-patch_width-20)
        y = random.randrange(p, Im.shape[0]-p-patch_height-20)

        patch = np.array([[x,y],[x+patch_width,y],[x+patch_width,y+patch_height],[x,y+patch_height]], dtype=np.float32)
        perturbed_patch = []

        # perform a random perturbation on the patch in the range [-p,p]
        for corner in patch:
            perturbed_patch.append([corner[0]+random.randrange(-p,p),corner[1]+random.randrange(-p,p)])
        perturbed_patch = np.array(perturbed_patch,dtype =np.float32)

        #Estimate Homography
        H_AB = cv.getPerspectiveTransform(patch,perturbed_patch)  # Homography matrix
        H_BA = np.linalg.inv(H_AB)   # And its Inverse

        # warp the image
        warpped_image = cv.warpPerspective(Im, H_BA, (Im.shape[1],Im.shape[0]))

        #extract the patches from warpped image
        cropped_A = Im[y:y+patch_height, x:x+patch_width]
        cropped_B = warpped_image[y:y+patch_height, x:x+patch_width]
        cropped_A = cropped_A.astype(np.double)
       
        # initializa modelparameters
        # generate dataset
        data = np.stack((cropped_A,cropped_B))

        h4pt = np.subtract(perturbed_patch,patch)
        h4pt = h4pt.reshape(8,-1)
        # restricting to the range (-1,1)
        h4pt = h4pt/32
        # Append all images and mask
        ImBatch.append(data)
        LabelBatch.append(h4pt)
        Im = cv.resize(Im,(128,128))
        Images.append(np.float32(Im))
        Patches.append(data)
        Corners.append(np.float32(patch))
        cropb.append(np.float32(cropped_B.reshape(128,128,1)))
    return ImBatch, LabelBatch, Images, Corners, cropb

    
    
def main():

    os.chdir("../Data/Train")
    k = Generate_Train_Dataset()
    os.chdir("../")
    # Cropped A and B
    np.save('ImBatch_.npy',np.array(k[0],dtype='float32'))
    #labelbatch h4pt
    np.save('LabelBatch_.npy',np.array(k[1],dtype='float32'))
    # #Original images
    # np.save('Images_.npy',np.array(k[2],dtype='float32'))
    #Corners
    np.save('Corners_.npy',np.array(k[3],dtype='float32'))
    # # crop B reshaped
    # np.save('Cropb_.npy',np.array(k[4],dtype='float32'))
    

    # Validation dataset
    os.chdir("../Data/Val")
    k = Generate_Test_Dataset()
    os.chdir("../")
    # Cropped A and B
    np.save('ImBatch_val.npy',np.array(k[0],dtype='float32'))
    #labelbatch h4pt
    np.save('LabelBatch_val.npy',np.array(k[1],dtype='float32'))
    #Original images
    np.save('Images_val.npy',np.array(k[2],dtype='float32'))
    #Corners
    np.save('Corners_val.npy',np.array(k[3],dtype='float32'))
    # # crop B reshaped
    # np.save('Cropb_.npy',np.array(k[4],dtype='float32'))


if __name__ == '__main__':
	main()
