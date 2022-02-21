

Title of the project:
In the script Panorama.py is a code that takes two images and stiches them into one panoramic image
We used in this script the Orb algorithm with Knnmatch and Homograph matrix in order to reach the result

1- first we take the images and resize them so they can be in the same height (big images are resized to 70% of the original size)
2- we turn them into gray scale in order for Orb algorithm to work
3- we detect keypoints in each image 
4- we use knn to match between the similar regions in both images
5- after the knn finshes we will have for each right image region two most close regions in the left image
after that we take the best of both by using  distance1<ratio*distance2
6- we take the key points of the best regions
7- we create a homography matrix and use a wraping function with right image
8- we add the left image and stech it to the final result
9- saving the panorama in the given path

Install Packages
  Chose “Terminal”at the bottom of the PyCharm
  Type “pip install opencv-contrib-python”.Note: it might take a few minutes to install the package
  Type “pip install matplotlib “ (I used it just to see the new images for once then i didnt used it , which means that i used it just for checking results)

After that you can import:

1) import cv2
2) import matplotlib.pyplot as plt (for me just to check resukts while im working in code)

you should also import:

3) import numpy as np
4) import argparse


how to run the code: to run the script enter a right image path , left image path and the path of where you want to save the panorama
  in terminal write:
                
            python Panorama.py '1/left.jpg' '1/right' '1/panora.jpg' 


  explain:
     Panorama.py       -> your code
     1/left.jpg    -> left image path for the panorama image
     1/right.jpg   -> right image path for the panorama image  
     1/panora.jpg  -> path that we use to save the panorama image 

