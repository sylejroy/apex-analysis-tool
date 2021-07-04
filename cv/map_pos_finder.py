import cv2

# x = coordinates to the right, y = coordinates down
MAP_CROP_TOP_LEFT_X = 55
MAP_CROP_TOP_LEFT_Y = 55
MAP_CROP_WIDTH = 230
MAP_CROP_HEIGHT = 230

def findMapPose(frame, reference):
    # Crop
    currentMap = frame[MAP_CROP_TOP_LEFT_Y:MAP_CROP_TOP_LEFT_Y + MAP_CROP_HEIGHT,
                 MAP_CROP_TOP_LEFT_X:MAP_CROP_TOP_LEFT_X + MAP_CROP_WIDTH]



    # edge detection
    #currentMap = cv2.Canny(currentMap, 100, 200)
    #reference = cv2.Canny(reference, 150, 200)

    cv2.imshow('current_frame', currentMap)
    cv2.imshow('reference', reference)#cv2.resize(reference, (1000, 1000)))

    method = cv2.TM_SQDIFF_NORMED
    result = cv2.matchTemplate(reference, currentMap, method)

    # We want the minimum squared difference
    mn, _, mnLoc, _ = cv2.minMaxLoc(result)

    print(mn)

    # Draw the rectangle:
    # Extract the coordinates of our best match
    MPx, MPy = mnLoc

    # Step 2: Get the size of the template. This is the same size as the match.
    trows, tcols = currentMap.shape[:2]

    # Step 3: Draw the rectangle on large_image
    cv2.rectangle(reference, (MPx, MPy), (MPx + tcols, MPy + trows), (0, 0, 255), 2)

    # Display the original image with the rectangle around the match.
    cv2.imshow('output', cv2.resize(reference, (1000, 1000)))

    cv2.waitKey()

    pos = [0, 0]

    return pos


