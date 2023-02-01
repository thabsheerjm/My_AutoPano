import torch 
import tqdm
import torch.nn as nn
import numpy as np
import kornia
from torchsummary import summary
from network import HomographyNet
import cv2 as cv
import copy
import torch.nn.functional as F
from sklearn import metrics 
import matplotlib.pyplot as plt

# Load the trained model
model = HomographyNet()
model.load_state_dict(torch.load('CNN_Homography_model.pth'))
model.eval()
#print(summary(model,(2,128,128)))
Loss_function = nn.MSELoss()
# Load test data (patch A and B)
tdata = np.load('ImBatch_val.npy')
tdata = tdata.astype(np.float32)
tdata = torch.tensor(tdata)

# Load test Labels h4pt (Ground truth)
tLabel = np.load('LabelBatch_val.npy')
tLabel = tLabel.astype(np.float32)
tLabel = torch.tensor(tLabel)

# LOad corners (CA) of patch A
CornerA = np.load('Corners_val.npy')
CornerA = CornerA.astype(np.float32)
CornerA = torch.tensor(CornerA).view(-1,1,8)

#Load original image
img = np.load('Images_val.npy')
img= img.astype(np.float32)
img = torch.tensor(img)

labelp = []

# test_set error
testset = 1000
out = model(tdata[0].view(-1,2,128,128))
loss = Loss_function(out,tLabel[0].view(-1,8))
with torch.no_grad():
    for i in range(0,testset):
        output = model(tdata[i].view(-1,2,128,128))
        loss = np.array(Loss_function(output, tLabel[i].view(-1,8)))
    print('\Mean Squared Error: {:.6f}'.format(np.mean(loss)))


# #Test a random image
# idx =27
# Patches = tdata[idx].view(-1,2,128,128)
# label = tLabel[idx].view(-1,8)
# IA = img[idx].view(1,1,128,128)
# PA = Patches.view(-1,128,128)[0]
# PA = PA.view(1,1,128,128)
# CA = CornerA[idx]

# CA = CA - CA[:, 0]  # Subtracting TOPLEFT corner

# h4pt_hat = model(Patches)
# # h4pt = CB - CA
# CB = h4pt_hat+CA
# H_matrix = kornia.geometry.homography.find_homography_dlt(CA.view(-1,4,2),CB.view(-1,4,2))
# h_inv = torch.inverse(H_matrix)

# PB_hat = kornia.geometry.transform.warp_perspective(PA,h_inv, dsize=(128, 128))
# Patches = Patches.view(2,128,128)
# PB = Patches[1]
# with torch.no_grad():
#     Pb = np.array(PB.view(128,128))
#     Pb_hat = np.array(PB_hat.view(128,128))
#     Pa = np.array(PA.view(128,128))

# cv.imshow('Patch A',Pa)
# cv.imshow('Patch B',Pb)
# cv.imshow('warped  Patch B',Pb_hat)
# cv.waitKey()
# cv.destroyAllWindows


def photometric_loss(model,Patches,CA):
    '''input : model
               Patches (PA and PB)
               CA Corner of image 1
        out: Loss
    '''
    Patches= Patches.view(-1,2,128,128)
    h4pt_hat = model(Patches)
    labelp.append(h4pt_hat)
    PA = Patches.view(-1,128,128)[0]
    PA = PA.view(-1,1,128,128)
    PB = Patches.view(-1,128,128)[1]
    PB = PB.view(-1,1,128,128)
    CB = h4pt_hat+CA
    CA = CA - CA[:, 0] 
    H_matrix = kornia.geometry.homography.find_homography_dlt(CA.view(-1,4,2),CB.view(-1,4,2))
    h_inv = torch.inverse(H_matrix)
    PB_hat = kornia.geometry.transform.warp_perspective(PA,h_inv, dsize=(128, 128))
    return F.l1_loss(PB_hat, PB)


for i in range(1000):
    loss = photometric_loss(model,tdata[i],CornerA[i])
print(loss)



# '''Confusion Matrix'''
# confusion_matrix = metrics.confusion_matrix(tLabel, labelp)
# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
# cm_display.plot()
# plt.show() 
