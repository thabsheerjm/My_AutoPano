from tracemalloc import start
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
from network import HomographyNet
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import time
import kornia
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #No Gpu

#Call the dataset

# For unsupervised learning use photometric loss function
# Tesnor DLT  is implemented inside
def photometric_loss(h4pt_hat,Patches,CA):
    '''input : model
               Patches (PA and PB)
               CA Corner of image 1
        out: Loss
    '''
    Patches= Patches.view(-1,2,128,128)
    PA = Patches.view(-1,128,128)[0]
    PA = PA.view(-1,1,128,128)
    PB = Patches.view(-1,128,128)[1]
    PB = PB.view(-1,1,128,128)
    CB = h4pt_hat+CA
    CA = CA - CA[:, 0] 
    H_matrix = kornia.geometry.homography.find_homography_dlt(CA.view(-1,4,2),CB.view(-1,4,2))
    h_inv = torch.linalg.pinv(H_matrix)
    PB_hat = kornia.geometry.transform.warp_perspective(PA,h_inv, dsize=(128, 128))
    return F.l1_loss(PB_hat, PB)


# Call the model
model = HomographyNet()
#print(summary(model,(2,128,128)))

# Load input data (patch A and B)
data = np.load('ImBatch_.npy')
data = data.astype(np.float32)
data = torch.tensor(data)

# Load Labels h4pt
Label = np.load('LabelBatch_.npy')
Label = Label.astype(np.float32)
Label = torch.tensor(Label)

# Load corners of image 1
corners = np.load('Corners_.npy')
corners = corners.astype(np.float32)
corners = torch.tensor(corners).view(-1,1,8)

# Helper functions

def tic():
    """
    Function to start timer
    Tries to mimic tic() toc() in MATLAB
    """
    StartTime = time.time()
    return StartTime

def toc(StartTime):
    """
    Function to stop timer
    Tries to mimic tic() toc() in MATLAB
    """
    return time.time() - StartTime


def TrainOperation(trainset,batch_size, epochs, loss_fn, optimizer):
    start = tic()
   
    for epoch in tqdm(range(epochs)):
        for i in tqdm(range(0,trainset,batch_size)):
            batch_x = data[i:i+batch_size]
            batch_y = Label[i:i+batch_size].view(-1,8)

            model.zero_grad()
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
        print("Epoch:", epoch,"Loss:",loss)
    end = toc(start)
    print("Time to train:", round(end,3)," seconds")
    #state = {'epoch':epochs, 'state_dcit':model.state_dict(),'optimizer':optimizer.state_dict()}
    torch.save(model.state_dict(),'CNN_Homography_model.pth')
    
    return model

# parameters

optimizer = optim.Adam(model.parameters(),lr=0.0001)

Num_trainset = 5000
Batch_size = 15
EPOCHS = 20
loss =  nn.MSELoss()
train = TrainOperation(Num_trainset,Batch_size, EPOCHS,loss, optimizer)



# '''uNSUPERVISED LEARNING'''

# def TrainOperation(trainset,batch_size, epochs, loss_fn, optimizer):
#     start = tic()
   
#     for epoch in tqdm(range(epochs)):
#         for i in tqdm(range(0,trainset,batch_size)):
#             batch_x = data[i:i+batch_size]
#             batch_y = Label[i:i+batch_size].view(-1,8)
#             cornersA = corners[i:i+batch_size]
#             model.zero_grad()
#             outputs = model(batch_x)
#             loss = photometric_loss(outputs,batch_x,cornersA)
#             loss.backward()
#             optimizer.step()
#         print("Epoch:", epoch,"Loss:",loss)
#     end = toc(start)
#     print("Time to train:", round(end,3)," seconds")
#     #state = {'epoch':epochs, 'state_dcit':model.state_dict(),'optimizer':optimizer.state_dict()}
#     torch.save(model.state_dict(),'CNN_Homography_model.pth')
    
#     return model

# # parameters

# optimizer = optim.Adam(model.parameters(),lr=0.0001)

# Num_trainset = 5000
# Batch_size = 50
# EPOCHS = 1
# loss =  nn.MSELoss()
# train = TrainOperation(Num_trainset,Batch_size, EPOCHS,loss, optimizer)

# # if __name__=='main':
# #     TrainOperation()

