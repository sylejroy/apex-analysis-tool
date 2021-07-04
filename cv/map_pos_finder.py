import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# x = coordinates to the right, y = coordinates down
CROP = 0
MAP_CROP_TOP_LEFT_X = 55 + CROP
MAP_CROP_TOP_LEFT_Y = 55 + CROP
MAP_CROP_WIDTH = 230 - CROP * 2
MAP_CROP_HEIGHT = 230 - CROP * 2

SCALE_RATIO = 1.85


def findMapPoseSIFT(frame, refMap, plotMatching=True, printTimer=False):
    # Start timer
    start = time.time()

    # Crop
    miniMap = frame[MAP_CROP_TOP_LEFT_Y:MAP_CROP_TOP_LEFT_Y + MAP_CROP_HEIGHT,
                    MAP_CROP_TOP_LEFT_X:MAP_CROP_TOP_LEFT_X + MAP_CROP_WIDTH]

    # FLANN based SIFT matching
    feature = cv2.BRISK_create()

    # Find the keypoints and descriptors with SIFT
    img1 = cv2.cvtColor(refMap, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(miniMap, cv2.COLOR_BGR2GRAY)

    kp1, des1 = feature.detectAndCompute(img1, None)
    kp2, des2 = feature.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Find pose from the matches
    pos = findPoseFromFeatureMatches(matches[:10], kp1, kp2)

    # End timer
    end = time.time()
    if printTimer:
        print(end-start)

    # Display SIFT matching
    if plotMatching:
        # Draw first 10 matches
        output = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('output', cv2.resize(output, (1000, 1000)))
        cv2.waitKey(1)


def findPoseFromFeatureMatches(matches, kp1, kp2):
    # img1 -> reference map
    # img2 -> minimap on current frame
    list_kp1 = []
    list_kp2 = []
    distance_array_kp1 = []
    distance_array_kp2 = []
    ratio = []
    index = 0

    for match in matches:
        img1Idx = match.queryIdx
        img2Idx = match.trainIdx
        (x1, y1) = kp1[img1Idx].pt
        (x2, y2) = kp2[img2Idx].pt
        list_kp1.append((x1, y1))
        list_kp2.append((x2 - MAP_CROP_WIDTH / 2, y2 - MAP_CROP_WIDTH / 2))

    # Find scale difference
    for idxRef, ptRef in enumerate(list_kp1):
        for idxTest, ptTest in enumerate(list_kp1):
            if idxRef != idxTest:
                distance_array_kp1.append((ptRef[0] - ptTest[0]) ** 2 + (ptRef[1] - ptTest[1]) ** 2)

    for idxRef, ptRef in enumerate(list_kp2):
        for idxTest, ptTest in enumerate(list_kp2):
            if idxRef != idxTest:
                distance_array_kp2.append((ptRef[0] - ptTest[0]) ** 2 + (ptRef[1] - ptTest[1]) ** 2)
                ratio.append(distance_array_kp1[index] / distance_array_kp2[index])
                index = index + 1

    scale_ratio = sum(ratio) / len(ratio)
    scale_ratio = max(set(ratio), key=ratio.count)
    print(scale_ratio)


    pos = [0, 0]
    return pos
