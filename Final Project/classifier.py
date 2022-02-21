'''
image processing and computer vision final project

Ibrahim Wattaweda - 207154212
Odai Wattad - 314821943

'''

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

testData = []
validData = []
trainData = []
total_time = None
start_time = None

kernel_models = [[1, 8, 'linear'], [1, 8, 'rbf'],  [3, 24, 'linear'], [3, 24, 'rbf']]

# PART I Loading dataset
def loadDataset(train, val, test):
    for gender in os.listdir(train):
        for img in os.listdir(train+"/"+gender):
            trainData.append([img, gender])

    for gender in os.listdir(val):
        for img in os.listdir(val + "/" + gender):
            validData.append([img, gender])

    for gender in os.listdir(test):
        for img in os.listdir(test + "/" + gender):
            testData.append([img, gender])

    # print(validData, " \n",testData)
'''
returns the best parameters for RBF algorithm
'''
def GridSearch_tuning(X_train, y_train):
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}
    # Make grid search classifier
    clf_grid = GridSearchCV(svm.SVC(kernel="rbf"), param_grid, verbose=1)
    # Train the classifier
    clf_grid.fit(X_train, y_train)
    # extract the parameters
    best_params = clf_grid.best_params_
    # return the parameters
    return best_params

def details():
    print('Image processing and Computer vision Course')
    print('Final Project of the course')
    print('Gender classification from handwritten documents using SVM(Support Vector Machine)')
    print('The project was done by Odai Wattad - 314821943 and Ibrahim Wattaweda - 207154212')
    print('')

# labels
def get_image_lables(imageData, flag):
    images = []
    labels = []
    dir=""
    if flag == 1:
        dir = TRAIN
    elif flag == 0:
        dir = TEST
    else :
        dir = VALIDATION

    for i in imageData:
        img = cv2.imread(dir + "/" +  i[1] + "/" + i[0])
        images.append(img)
        labels.append(i[1])

    return images, labels


# PART II - Feature extraction using  LBP
def LBPfeatures(images, radius, pointNum):
    hist_LBP = []
    for img in images:
        # print(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if type(pointNum) == int and type(radius) == int:
            lbp = feature.local_binary_pattern(gray, int(pointNum), int(radius), method="uniform")
            (hist, _) = np.histogram(lbp.ravel(), bins=range(0, pointNum + 3), range=(0, pointNum + 2))
            eps = 1e-7
            hist = hist.astype("float")
            hist /= (hist.sum() + eps)
            hist_LBP.append(hist)
    return hist_LBP


def extract_data(radius, points):
    # train data
    train_image, train_label = get_image_lables(trainData, 1)
    # Validation data
    valid_image, valid_label = get_image_lables(validData, 0)

    # Feature Extraction using LBP
    train_features = callLBP(train_image, radius, points)
    valid_features = callLBP(valid_image, radius, points)

    # return train and validation images with labels.
    return train_image, train_label,valid_image, valid_label,train_features,valid_features

# PART III - training image
def train(kernel):
    # extracting kernel data
    radius, points, ker = kernel
    # all the data that we need
    train_image, train_label, valid_image, valid_label, train_features, valid_features = extract_data(radius, points)

    if ker == 'rbf':
        params = GridSearch_tuning(train_features, train_label)
        model = SVC(kernel=ker, C=params['C'], gamma=params['gamma'])
        model.fit(train_features, train_label)
        model_predictions = model.predict(valid_features)
        accuracy = accuracy_score(valid_label, model_predictions)

    else:
        # model train:
        model = SVC(kernel = ker)
        model.fit(train_features, train_label)
        # validation on
        model_predictions = model.predict(valid_features)
        # calculating model accuracy
        accuracy = accuracy_score(valid_label, model_predictions)
    return accuracy, ker, model

def callLBP(images, points, rads):
    test = LBPfeatures(images,points,rads)
    return test

def write_results(best_E, accuracy, CM,best_points,best_rad):
    f = open("results.txt", "w")
    ker = best_E
    f.write('Ibrahim And Odai results')
    f.write("model parameters :\nkernel= {0}, Number of points= {1}, Radius= {2} \n".format(ker, best_points, best_rad))
    f.write("Accuracy : {:.2f}%".format(accuracy * 100))
    f.write('\nConfusion matrix:')
    f.write(' ' * 13 + 'male' + '  ' + 'female')
    f.write('\nmale' + ' ' * 7 + str(CM[0][0]) + ' ' * 4 + str(CM[0][1]))
    f.write('\nfemale' + ' ' * 3 + str(CM[1][0]) + ' ' * 4 + str(CM[1][1]))
    time1 = round(time.time() - start_time)
    timer = time.strftime('%H:%M:%S', time.gmtime(time1))
    f.write('\nTime took to train SVM : ' + timer)
    f.close()

# main
if __name__ == "__main__":
    details()
    start_time = time.time()
    best_kernel = None
    Best_acc = -1
    Best_model = None
    best_rad = None
    best_points = None
    test_features = []
    if len(sys.argv) < 2 :
        print('the code only runs from command line')
        print('if you do not know how to run simply write in the command prompt "python classifier.py --help" and it will show you the instructions  ')
    else:
        if sys.argv[1] == '--help' or sys.argv[1] == '-h':
            print('list of params that the code needs are \n1.train folder\n2.test folder\n3.validation folder\n')
            print('run command must be in this format: python classifier.py train_dir test_dir val_dir\n')
            print('if any of the directories are missing the program will not work.')
        else:
            if len(sys.argv) < 4:
                print('\nnot all directories supported to the program\n')
                print('run command must be in this format: python classifier.py train_dir test_dir val_dir\n')
            else:
                if loadDataset(sys.argv[1], sys.argv[2], sys.argv[3]) != 0:
                    TRAIN = sys.argv[1]
                    VALIDATION = sys.argv[3]
                    TEST = sys.argv[2]
                    for i in kernel_models:
                        accuracy, kernel, model = train(i)
                        if accuracy > Best_acc:
                            Best_acc = accuracy
                            best_kernel = kernel
                            best_rad = i[1]
                            best_points = i[0]
                            Best_model = model
                    test_images, test_labels = get_image_lables(testData, 3)
                    test_features = callLBP(test_images, best_points, best_rad)
                    model_prediction = Best_model.predict(test_features)
                    accuracy = accuracy_score(test_labels, model_prediction)
                    model_confusion_matrix = confusion_matrix(test_labels, model_prediction)
                    write_results(best_kernel, Best_acc, model_confusion_matrix, best_points, best_rad)