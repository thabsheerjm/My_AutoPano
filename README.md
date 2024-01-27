# My AutoPano - Image Stitching 
In this project we are learning to stitch two or more images in order to create a seamless panorama by finding the homohraphy between the two images. 

## Phase 1 : Classical Computer Vision approach 
The claasical method involves several steps, it starts with corner detection. Then Adaptive  Non-maximal Suppression (ANMS) is applied to ensure an even distribution of conrners. Feature descriptors are created by encoding information at each feature point into a vector. The next step is feature matching, where feature points between images are matched. RANSAC is used for outlier rejection and to estimate a robust homography. Finally, the images are blended to produce the panorama.The deatiled steps are explained [here](https://rbe549.github.io/fall2022/proj/p1/).  

### How to Run:
Place the image pairs in the dir:"../Data/Train/Set1/" and run the following command:  
```
 Wrapper.py
```

The classical pipeline is given below:  
<img src="Results/p1.png"  align="center" alt="Undistorted" width="500"/>

### Corner Detection
<img src="Results/p1_corners.png"  align="center" alt="Undistorted" width="500"/>  

### Non-Maximal Suppression
<img src="Results/p1_anms.png"  align="center" alt="Undistorted" width="500"/>

### Feature Matching 
<img src="Results/p1_featurematching.png"  align="center" alt="Undistorted" width="500"/>

### Outlier Rejection
<img src="Results/p1_ransac.png"  align="center" alt="Undistorted" width="500"/>

### Parorama: Warping, Blending and Stitching
<img src="Results/pano.png"  align="center" alt="Undistorted" width="500"/>

## Phase 2: Deep Learning
The deep learning model effectively combines corner detection, ANMS, feature extraction, feature matching, RANSAC and estimate homography all into one robust generalizable network. The complete methodolgy is given [here](https://rbe549.github.io/fall2022/proj/p1/). 

### How to Run:

Run the following to generate the dataset
```
Datagen.py
```
Divide train and validation. and run:
```
network.py
Train.py
Test.py
```

The network pipeline is given below:  
<img src="Results/p2.png"  align="center" alt="Undistorted" width="500"/>

We followed a supervised model and obtained the follwing results:

The following picture shows the input and output patch from the trained network. 
<img src="Results/p2_res.png"  align="center" alt="Undistorted" width="500"/>

The training loss over number of epochs is given below:

<img src="Results/p2_loss.png"  align="center" alt="Undistorted" width="500"/>

## Acknowledgement 

This project was part of RBE549- Computer Vision (Fall 22) at Worcester Polytechic Institute[[1]](https://rbe549.github.io/fall2022/proj/p1/).  
Team members :[Thabsheer Machingal](https://github.com/thabsheerjm) and [Krishna Madhurkar](https://github.com/Krishnamadhurkar)

## References 
[1] [RBE549-Project1](https://rbe549.github.io/fall2022/proj/p1/)
