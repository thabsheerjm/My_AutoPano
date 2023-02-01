Project 1: My Autopano

Phase 1:
Pahse 1 code is located at: tmachingal&kmadhurkar_p1/Phase1/Code/Wrapper.py
Phase 2 code is located at: tmachingal&kmadhurkar_p1/Phase2/Code/

Phase2 code : Three required files:
                                     'Dataset.py' --> Create the synthetic dataset and save it as .npy and later is loaded to 'Train.py' and 'Test.py' 
                                     'network.py' --> Has the HomographyNet() cnn model
                                     'Train.py'   --> Train the model and save the model as .pth file and later load the model to 'Test.py'
                                     'Test.py'    --> Test the images from validation set
                                     

How to run:
Phase 1: Run Wrapper.py and place the image pairs in the dir:"../Data/Train/Set1/"

Phase2: Run 'Datagen.py' first to generate dataset for which place training data at dir: "../Data/Train" and validation data at dir:"../Data/Val"
        Then run 'network.py', 'Train.py' and 'Test.py'
        
        
       To test a random image just uncomment line 53 to 82 on 'Test.py'
       
