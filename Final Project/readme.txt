
Final Project in course Image processing and computer vision


Title of the project:
    Gender classification from handwritten images


Install Packages
  Chose “Terminal” at the bottom of the PyCharm
  Type “pip install opencv-contrib-python”.Note: it might take a few minutes to install the package
  Type "pip install -U scikit-image"
  Type "pip install -U scikit-learn"

After that you must import:
    import os
    import sys
    import time
    import numpy as np
    from skimage import feature
    import cv2
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.svm import SVC
    from sklearn import svm
    from sklearn.model_selection import GridSearchCV

how to run the code:
    write the following command : python classifier.py train_dir test_dir val_dir
  explain:
     classifier.py -> code script
     train_dir    -> folder that contains training images
     test_dir   -> folder that contains testing images
     val_dir  -> folder that contains validation images

results.txt contains the best results that the classifier got.

you can run the following command to know more about script run :
    "python classifier.py --help" or "python classifier.py -h"

