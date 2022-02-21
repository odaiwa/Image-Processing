import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


def Scanner(img):

    GrayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    BLurredFrame = cv2.GaussianBlur(GrayImg, (5, 5), 1)
    ret1, th1 = cv2.threshold(BLurredFrame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    CannyFrame = cv2.Canny(th1, 190, 190)
    contours, _ = cv2.findContours(CannyFrame,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    ContourFrame = img.copy()
    ContourFrame = cv2.drawContours(ContourFrame, contours, -1, (0, 255, 0), 20)
    CornerFrame = img.copy()
    maxArea = 0
    biggest = []
    for i in contours :
        area = cv2.contourArea(i)
        peri = cv2.arcLength(i, True)
        edges = cv2.approxPolyDP(i, 0.02 * peri, True)
        if area > maxArea and len(edges) == 4:
            biggest = edges
            maxArea = area
    if len(biggest) != 0:
        biggest = biggest.reshape((4, 2))
        biggestNew = np.zeros((4, 1, 2), dtype=np.int32)
        add= biggest.sum(1)
        biggestNew[0] = biggest[np.argmin(add)]
        biggestNew[3] = biggest[np.argmax(add)]
        dif = np.diff(biggest, axis=1)
        biggestNew[1] = biggest[np.argmin(dif)]
        biggestNew[2] = biggest[np.argmax(dif)]
        CornerFrame = cv2.drawContours(CornerFrame, biggestNew, -1, (0, 255, 0), 25)
        (h,w,d)=CornerFrame.shape
        pts1 = np.float32(biggestNew)
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
        WarpImg=cv2.warpPerspective(img, matrix, (w, h))

    return WarpImg


def iterator(images_path, output_path):
    # create output folder (it will be create on the same directory that we are on only if full path supported)
    i = 1
    if (not os.path.exists(output_path)):
        os.makedirs(output_path)
    else:
        print("Directory {0} already exists".format(output_path))
    images = os.listdir(images_path)  # returns list

    for o in images:
        # 1- reading the image
        image = cv2.imread('./'+images_path+'/' + o)
        image = cv2.resize(image, (int(520 * 2), int(670 * 2)))
        After_Scan = Scanner(image)
        new_image = "./" + output_path + "/New_"
        print(new_image)
        print(output_path)
        print(i)
        cv2.imwrite(new_image+i.__str__()+".jpg", After_Scan)
        i = i+1


if __name__ == "__main__":
    iterator(sys.argv[1], sys.argv[2])
