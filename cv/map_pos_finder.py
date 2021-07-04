import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# x = coordinates to the right, y = coordinates down
CROP = 30
MAP_CROP_TOP_LEFT_X = 55 + CROP
MAP_CROP_TOP_LEFT_Y = 55 + CROP
MAP_CROP_WIDTH = 230 - CROP * 2
MAP_CROP_HEIGHT = 230 - CROP * 2


def findMapPoseSIFT(frame, refMap, plotMatching=True):
    # Start timer
    start = time.time()

    # Crop
    miniMap = frame[MAP_CROP_TOP_LEFT_Y:MAP_CROP_TOP_LEFT_Y + MAP_CROP_HEIGHT,
                    MAP_CROP_TOP_LEFT_X:MAP_CROP_TOP_LEFT_X + MAP_CROP_WIDTH]

    # Convert to gray scale
    refMap = cv2.cvtColor(refMap, cv2.COLOR_BGR2GRAY)
    miniMap = cv2.cvtColor(miniMap, cv2.COLOR_BGR2GRAY)

    # FLANN based SIFT matching
    feature = cv2.BRISK_create()

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = feature.detectAndCompute(refMap, None)
    kp2, des2 = feature.detectAndCompute(miniMap, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Find pose from the matches
    pos = findPoseFromFeatureMatches(matches)

    # End timer
    end = time.time()
    print(end-start)

    # Display SIFT matching
    if plotMatching:
        # Draw first 10 matches.
        output = cv2.drawMatches(refMap, kp1, miniMap, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('output', cv2.resize(output, (1000, 1000)))
        cv2.waitKey(1)


def findPoseFromFeatureMatches(matches):
    matches

    pos = [0, 0]
    return pos


# def findMapPose(frame, reference):
#     # Crop
#     currentMap = frame[MAP_CROP_TOP_LEFT_Y:MAP_CROP_TOP_LEFT_Y + MAP_CROP_HEIGHT,
#                        MAP_CROP_TOP_LEFT_X:MAP_CROP_TOP_LEFT_X + MAP_CROP_WIDTH]
#
#     method = cv2.TM_SQDIFF_NORMED
#
#     scale = np.linspace(1.0, 1.6, num=20)
#
#     best_score = 0
#     best_loc = [0, 0]
#
#     for ratio in scale:
#         width = int(currentMap.shape[1] * ratio)
#         length = int(currentMap.shape[0] * ratio)
#         dim = (width, length)
#
#         resizedMiniMap = cv2.resize(currentMap, dim, interpolation=cv2.INTER_NEAREST)
#
#         result = cv2.matchTemplate(reference, resizedMiniMap, method)
#
#         # We want the minimum squared difference
#         mn, _, mnLoc, _ = cv2.minMaxLoc(result)
#
#         print('Minimum square distance = ' + str(mn) + ' With a scale of ' + str(ratio))
#
#         if mn > best_score:
#             best_score = mn
#             best_resized = resizedMiniMap
#             best_loc = mnLoc
#
#     # Draw the matched rectangle:
#     # Extract the coordinates of our best match
#     MPx, MPy = best_loc
#     currentMap = best_resized
#     # Step 2: Get the size of the template. This is the same size as the match.
#     trows, tcols = currentMap.shape[:2]
#
#     # Step 3: Draw the rectangle on large_image
#     cv2.rectangle(reference, (MPx, MPy), (MPx + tcols, MPy + trows), (0, 0, 255), 2)
#
#     # Display cropped map and reference
#     cv2.imshow('current_frame', currentMap)
#     cv2.imshow('reference', reference)
#
#     # Display the original image with the rectangle around the match.
#     cv2.imshow('output', reference)
#     cv2.imshow('output', cv2.resize(reference, (1000, 1000)))
#
#     cv2.waitKey()
#
#     pos = [0, 0]
#
#     return pos
