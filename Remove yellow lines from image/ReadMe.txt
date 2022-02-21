
Title of the project:
In this project we have pictures with writing on yellow colored lines 
and the goal is to remove the yellow lines so that we get a picture with writing only.
(it should be noted that the writing is not in yellow, but the lines only) 

Install Packages
  Chose “Terminal”at the bottom of the PyCharm
  Type “pip install opencv-contrib-python”.Note: it might take a few minutes to install the package
  Type “pip install matplotlib “ (I used it just to see the new images for once then i didnt used it , which means that i used it just for checking results)

After that you can import:

1) import cv2
2) import matplotlib.pyplot as plt

you should also import:

3) import sys
4) import numpy as np
5) import os

how to run the code:
  in terminal write:
                
            python hw1.py text_regions output1


  explain:
     hw1.py       -> your code
     text_regions -> input folder (scan folder)
     output1      -> the folder that we want to save the images inside

