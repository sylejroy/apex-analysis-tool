import cv2
import numpy as np
import time

from pykalman import KalmanFilter

# x = coordinates to the right, y = coordinates down

# Minimap cropping parameters
ADDITIONAL_CROP = 0
MAP_CROP_TOP_LEFT_X = 55 + ADDITIONAL_CROP
MAP_CROP_TOP_LEFT_Y = 55 + ADDITIONAL_CROP
MAP_CROP_WIDTH = 230 - ADDITIONAL_CROP * 2
MAP_CROP_HEIGHT = 230 - ADDITIONAL_CROP * 2

SCALE_FOR_BRISK = 0.5  # Scaling ratio to be applied to the reference map
SCALE_RATIO = 1.81 * SCALE_FOR_BRISK  # Approximate scale difference between minimap and reference map

# Matching parameters
NUM_MATCH_POS_EST = 3


def findMapPoseBRISK(frame, refMap, visu, log=False, cropRef=False, x_est=0, y_est=0):
    # Start timer
    if log:
        start = time.time()

    # Crop frame to contain only minimap
    miniMap = frame[MAP_CROP_TOP_LEFT_Y:MAP_CROP_TOP_LEFT_Y + MAP_CROP_HEIGHT,
                    MAP_CROP_TOP_LEFT_X:MAP_CROP_TOP_LEFT_X + MAP_CROP_WIDTH]

    # FLANN based SIFT matching
    feature = cv2.BRISK_create()

    # Crop reference map
    cropped = False
    if cropRef:
        refMap = refMap[int(y_est) - 200: int(y_est) + 200, int(x_est) - 200: int(x_est) + 200]
        cropped = True

    # Scale reference map
    refMap = cv2.resize(refMap, (int(refMap.shape[0] * SCALE_FOR_BRISK), int(refMap.shape[1] * SCALE_FOR_BRISK)))

    # Find the keypoints and descriptors with SIFT
    img1 = cv2.cvtColor(refMap, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(miniMap, cv2.COLOR_BGR2GRAY)
    kp1, des1 = feature.detectAndCompute(img1, None)
    kp2, des2 = feature.detectAndCompute(img2, None)

    # Check for empty descriptors
    if des2 is None or des1 is None:
        return [-1, -1]

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance (akin to goodness of fit)
    matches = sorted(matches, key=lambda x: x.distance)

    # Find pose from the matches
    measPose = findPoseFromFeatureMatches(matches[:NUM_MATCH_POS_EST], kp1, kp2)

    # End timer
    if log:
        end = time.time()
        print(end-start)

    # Display SIFT matching
    if visu:
        cv2.circle(img1, (int(measPose[0]), int(measPose[1])), 5, (255, 255, 255), 4)
        # Draw first x matches
        output = cv2.drawMatches(img1, kp1, img2, kp2, matches[:NUM_MATCH_POS_EST], None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('output', cv2.resize(output, (1000, 1000)))
        cv2.waitKey(1)

    pos = measPose / SCALE_FOR_BRISK

    # f = open('pose_estimation.csv', 'a')
    # f.write(str(pos[0]) + ';' + str(pos[1]) + '\n')

    if cropped:
        pos = [pos[0] + x_est - 200, pos[1] + y_est-200]

    return pos


def findPoseFromFeatureMatches(matches, kp1, kp2, printPos=False):
    # img1 -> reference map
    # img2 -> minimap on current frame
    list_kp1 = []
    list_kp2 = []
    distance_array_kp1 = []
    distance_array_kp2 = []
    ratio = []
    index = 0
    list_pos_estimate = []

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

    # scale_ratio = max(set(ratio), key=ratio.count)
    # if abs(scale_ratio / SCALE_FOR_BRISK - SCALE_RATIO) / SCALE_RATIO > 0.2:
    #     print(scale_ratio)
    #
    # f = open('scale_ratio.csv', 'a')
    # f.write(str(scale_ratio) + '\n')

    for idx, ptMini in enumerate(list_kp2):
        list_pos_estimate.append(np.array(list_kp1[idx]) - (np.array(ptMini) * SCALE_RATIO))

    pos = sum(list_pos_estimate) / len(list_pos_estimate)
    if printPos:
        print(pos)

    return pos
