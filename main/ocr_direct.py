import os
import shutil
import sys
import time
import math
import cv2
import numpy as np
import tensorflow as tf
import pyarabic.araby as araby
import string
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import keras.backend as K
from keras.models import load_model
from collections import Counter
from string import punctuation
letters = u'٠١٢٣٤٥٦٧٨٩'+'0123456789'


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def show_img(img, title="test"):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ocrline(line, model, letters):
    #line = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)
    line = cv2.resize(line, (432, 32))
    line = line/255.0
    line = np.expand_dims(line, -1)
    line = np.expand_dims(line, axis=0)
    prediction = model.predict(line)
    # use CTC decoder
    out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                                   greedy=True)[0][0])
    # see the results
    i = 0
    text = ''
    for x in out:
        print("predicted text = ", end='')
        for p in x:
            if int(p) != -1:
                try:
                    print(letters[int(p)], end='')
                    text += letters[int(p)]
                except IndexError:
                    pass
        print('\n')
        i += 1
    return text


if __name__ == "__main__":
    ocr_arch_path = 'nets/ocr/test_model.h5'
    ocr_weights_path = 'checkpoints_mlt/ocr/CRNN--50--0.025.hdf5'
    ocr = load_model(ocr_arch_path)
    ocr.load_weights(ocr_weights_path)
    for img in os.listdir('data/lines/'):
        im = cv2.imread('data/lines/'+img, 0)
        show_img(im, 'line')
        ocrline(im, ocr, letters)
