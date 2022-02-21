import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt

# arguments to take from console
parser = argparse.ArgumentParser(
    description="Making a panorama picture using a left and right picture to create it ...",
    epilog="And that's it :) ... ")
parser.add_argument('left_p', type=str, metavar='<left picture path>',
                    help="please enter a picture path to be on the left side of the panorama")
parser.add_argument('right_p', type=str, metavar='<right picture path>',
                    help="please enter a picture path to be on the right side of the panorama")
parser.add_argument('final_p', type=str, metavar='<the path you want to save the panorama at>',
                    help="please enter a path for the panorama to be saved at or just the"
                         " name of the file and it will be saved"
                         ".")
args = parser.parse_args()


def resize_image_by_height(left_img, right_img):
    """
    we resize the images while keeping the aspect ration intact
    :param left_img: the image we want to resize
    :param right_img: the image we use it's height to resize
    :return: return the left image resized
    the left and image is just for making the names easy, but left can be right and vice versa
    """
    left_height, left_width, _ = left_img.shape
    desired_height, _, _ = right_img.shape

    # taking the ratio of the original size of left image
    ratio = left_height / left_width

    # create new width using desire height and old ratio
    new_w = int(desired_height / ratio)

    # return the image with the new dims
    return cv2.resize(left_img, (new_w, desired_height))

def change_Nfutures(left_path):
    """
    :param left_path: image path
    :return: number of matches lines we want to compare between the two images
    """
    if (left_path[0] =='5') :
        return 6000
    return 500


def resize_image(img):
    """
    :arg img: image with big dimensions
    :returns:resized image
    """
    return cv2.resize(img, (0, 0), fx=0.7, fy=0.7)



def from_bgr_to_gray(img):
    """
    :arg img: BGR image
    :returns: a grayscale image
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def without_black(Panorama_image):
    """
    :param Panorama_image: panorama image with black area
    :return: panorama image without much black area
    """
    gray = cv2.cvtColor(Panorama_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = Panorama_image[y:y + h, x:x + w]
    return crop


def make_panorama(left_path, right_path, panorama_path, draw_match=False):
    """
    :param draw_match: if we want to to draw the matches after we detect them on the two pictures
    :arg left_path: an image path to be put on the left side of the panorama
    :arg right_path: an image path to put on the right side of the panorama
    :arg panorama_path: a path to save the panorama (which made of the two pictures)   
    """
    #reading the pictures
    left_image = cv2.imread(left_path)
    right_image = cv2.imread(right_path)

    #taking the height to check which picture has the biggest height
    left_height, w_l, _ = left_image.shape
    right_height, w_r, _ = right_image.shape

    # resizing images
    if left_height * w_l > 1000000:
        left_image = resize_image(left_image)
        left_height, _, _ = left_image.shape
    if right_height * w_r > 1000000:
        right_image = resize_image(right_image)
        right_height, _, _ = right_image.shape
    Nfeatures = change_Nfutures(left_path)


    # checking the height of each image to know what to resize
    if left_height > right_height:
        left_image = resize_image_by_height(left_image, right_image)
    elif left_height < right_height:
        right_image = resize_image_by_height(right_image, left_image)

    #convert images to gray scale in order to work with orb algorithm
    left_gray_pic = from_bgr_to_gray(left_image)
    right_gray_pic = from_bgr_to_gray(right_image)

    # using the orb algorithm
    orb = cv2.ORB_create(Nfeatures)
    keypointsLeft, descriptorsLeft = orb.detectAndCompute(left_gray_pic, None)
    keypointsRight, descriptorsRight = orb.detectAndCompute(right_gray_pic, None)

    bf = cv2.BFMatcher()

    raw_matches = bf.knnMatch(descriptorsRight, descriptorsLeft, k=2)

    # the ratio of distance because we get two neighbors
    ratio = 0.85

    good_matches = []

    for m1, m2 in raw_matches:
        if m1.distance < ratio * m2.distance:
            good_matches.append([m1])

    left_height, left_width = left_gray_pic.shape  # we take the height
    right_height, right_width = right_gray_pic.shape
    if draw_match:
        img_match = np.empty((max(left_height, right_height), left_width + right_width, 3),
                             dtype=np.uint8)
        imMatches = cv2.drawMatchesKnn(left_image, keypointsLeft, right_image, keypointsRight, good_matches, img_match, None)
        cv2.imshow('Matches between the two pictures', imMatches)

    # taking the key points of the two pictures
    right_image_kp = np.float32([keypointsRight[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    left_image_kp = np.float32([keypointsLeft[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # finding the homography matrix
    H, status = cv2.findHomography(right_image_kp, left_image_kp, cv2.RANSAC, 5.0)

    #warpPerspective the image
    Panorama_image = cv2.warpPerspective(right_image, H, (left_width + right_width, right_height))

    Panorama_image[0:left_image.shape[0], 0:left_image.shape[1]] = left_image

    Panorama_image= without_black(Panorama_image)
    cv2.imwrite(panorama_path, Panorama_image)
    print("Panorama image saved in [ " +panorama_path+ " ]")


if __name__ == '__main__':
    if args.left_p and args.right_p and args.final_p:
        make_panorama(args.left_p, args.right_p, args.final_p)
    else:
        print("In order to run the script you need to enter three arguments , use --help for more information")