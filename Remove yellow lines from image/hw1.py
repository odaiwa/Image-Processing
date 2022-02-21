import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


def Without_Yellow(txt_img):
    #  Reading and convert to HSV
    hsv = cv2.cvtColor(txt_img, cv2.COLOR_BGR2HSV)

    #  Finding the target yellow color region in HSV (from the internet)
    hsv_lower = np.array([21, 39, 64])
    hsv_upper = np.array([40, 255, 255])

    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    cv2.imwrite("mask.jpg", mask)
    src2 = cv2.imread('mask.jpg', cv2.IMREAD_COLOR)

    # blend the images
    no_yellow_img = cv2.addWeighted(txt_img, 1, src2, 1, 0.0)

    return no_yellow_img


def iterator(images_path, output_path):
    # create output folder (it will be create on the same directory that we are on only if full path supported)
    i = 13
    if (not os.path.exists(output_path)):
        os.makedirs(output_path)
    else:
        print("Directory {0} already exists".format(output_path))
    images = os.listdir(images_path)  # returns list

    for o in images:
        # 1- reading the image
        image = cv2.imread('./'+images_path+'/' + o)
        # 2- calling the func that removes the yellow lines
        hsv = Without_Yellow(image)
        # 3- prepair the new image path
        new_image = "./" + output_path + "/New_"
        print(new_image)
        print(output_path)
        print(i)
        # 4- write the new image (hsv(without yellow lines)) to the output directory
        cv2.imwrite(new_image+i.__str__()+".jpg", hsv)
        i = i+1


if __name__ == "__main__":
    iterator(sys.argv[1], sys.argv[2])
