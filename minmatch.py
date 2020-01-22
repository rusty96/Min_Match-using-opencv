# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:31:24 2020

@author: Suraj.Negi
"""
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10
img1 = cv.imread('sample1.jpg', 0)
img2 = cv.imread('sample.jpg', 0)

sift = cv.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)
good = []
for m,n in matches:
    if m.distances < 0.7*n.distance:
        good.append(m)
        
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    
    h,w,d = img1.shape
    pts = np.float32([0,0],[0,h-1],[w-1,h-1],[w-1,0]).reshape(-1,1,2)
    dst = cv.perpectiveTransform(pts, M)
    
    img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    
else:
    print("Not enough matches- {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None
    
draw_params = dict(matchColor = (0,255,0), singlePointColor=None,matchesMask = matchesMask, flags = 2)

img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
plt.imshow(img3, 'gray'),plt.show()
