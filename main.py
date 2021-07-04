import cv2
import numpy as np
from cv.map_pos_finder import findMapPoseBRISK

ROI_WIDTH = 400
ROI_HEIGHT = 400

# Video name
vid_path = 'data/vids/lifeline 4k.mp4'

# Read media
cap = cv2.VideoCapture(vid_path)
refMap = cv2.imread('data/map/we_map.png')

# Check if vid is opened successfully
if (cap.isOpened() == False):
    print("Unable to open video: " + vid_path)

index = 0
poseHistory = np.array([(-1, -1)])

# Read until video is finished
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    index = index + 1
    if (ret == True):
        # Display frame
        cv2.imshow(vid_path, frame)
        if (index % 400 == 0):
            # Find map pose estimate
            poseHistory = np.append(poseHistory, [findMapPoseBRISK(frame, refMap, poseHistory[-1])], 0)

            # Draw pose history as a series of lines
            for idx, pt in enumerate(poseHistory):
                if idx <= 1:
                    continue
                else:
                    cv2.line(refMap, (int(pt[0]), int(pt[1])), (int(poseHistory[idx-1][0]), int(poseHistory[idx-1][1])),
                             (0, 0, 255), 3)

            # Show pose estimation on reference map
            cv2.imshow('Pose estimation', cv2.resize(refMap, (1000, 1000)))

        # Press Q on keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release when done
cap.release()

# Close all frames
cv2.destroyAllWindows()