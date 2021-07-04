import cv2
import numpy as np

# x = coordinates to the right, y = coordinates down
CROP = 0
MAP_CROP_TOP_LEFT_X = 55 + CROP
MAP_CROP_TOP_LEFT_Y = 55 + CROP
MAP_CROP_WIDTH = 230 - CROP * 2
MAP_CROP_HEIGHT = 230 - CROP * 2

def findMapPose(frame, reference):
    # Crop
    currentMap = frame[MAP_CROP_TOP_LEFT_Y:MAP_CROP_TOP_LEFT_Y + MAP_CROP_HEIGHT,
                 MAP_CROP_TOP_LEFT_X:MAP_CROP_TOP_LEFT_X + MAP_CROP_WIDTH]

    method = cv2.TM_SQDIFF_NORMED

    scale = np.linspace(1.0, 1.6, num=20)

    best_score = 0
    best_loc = [0, 0]

    for ratio in scale:
        width = int(currentMap.shape[1] * ratio)
        length = int(currentMap.shape[0] * ratio)
        dim = (width, length)

        resizedMiniMap = cv2.resize(currentMap, dim, interpolation=cv2.INTER_NEAREST)

        #resizedMiniMap = cv2.blur(resizedMiniMap, (5, 5))
        #reference = cv2.blur(reference, (5, 5))

        result = cv2.matchTemplate(reference, resizedMiniMap, method)

        # We want the minimum squared difference
        mn, _, mnLoc, _ = cv2.minMaxLoc(result)

        print('Minimum square distance = ' + str(mn) + ' With a scale of ' + str(ratio))

        if mn > best_score:
            best_score = mn
            best_resized = resizedMiniMap
            best_loc = mnLoc

    # Draw the matched rectangle:
    # Extract the coordinates of our best match
    MPx, MPy = best_loc
    currentMap = best_resized
    # Step 2: Get the size of the template. This is the same size as the match.
    trows, tcols = currentMap.shape[:2]

    # Step 3: Draw the rectangle on large_image
    cv2.rectangle(reference, (MPx, MPy), (MPx + tcols, MPy + trows), (0, 0, 255), 2)

    # Display cropped map and reference
    cv2.imshow('current_frame', currentMap)
    cv2.imshow('reference', reference)

    # Display the original image with the rectangle around the match.
    cv2.imshow('output', reference)
    cv2.imshow('output', cv2.resize(reference, (1000, 1000)))

    cv2.waitKey()

    pos = [0, 0]

    return pos


