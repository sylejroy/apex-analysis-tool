import cv2
import numpy as np

from kalman_filter import KalmanFilter

from cv.pos_estimator import findMapPoseBRISK

ROI_WIDTH = 400
ROI_HEIGHT = 400

# Video name
vid_path = 'data/vids/lifeline 4k.mp4'

# Read media
cap = cv2.VideoCapture(vid_path)
refMap = cv2.imread('data/map/we_map.png')

# Check if vid is opened successfully
if not cap.isOpened():
    print("Unable to open video: " + vid_path)

index = 0
measHistory = np.array([])
estHistory = np.array([])

# Initialise KF
KF = KalmanFilter(0.5, 0, 0, 1, 30, 30)

# Read until video is finished
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    index = index + 1

    if ret:
        # Display frame
        cv2.imshow(vid_path, frame)

        if index % 30 == 0:  # Only run pose estimation on every xth frame
            refMapVis = refMap.copy()

            # Find map pose estimate
            if len(measHistory) == 0:
                measHistory = np.array([findMapPoseBRISK(frame, refMap, False, False)])
                KF.initState(int(measHistory[-1][0]), int(measHistory[-1][1]))
            else:
                measHistory = np.append(measHistory, [findMapPoseBRISK(frame, refMap, True, False)], 0)

            (x, y) = KF.predict()

            (x1, y1) = KF.update([measHistory[-1][0], measHistory[-1][1]])

            if len(estHistory) == 0:
                estHistory = np.array([[np.asarray(x1)[0][0], np.asarray(y1)[0][1]]])
            else:
                estHistory = np.append(estHistory, [np.array([np.asarray(x1)[0][0], np.asarray(y1)[0][1]])], 0)

            # Draw meas and estimated pose history
            for idx, pt in enumerate(np.flip(np.flip(measHistory, 0)[:10], 0)):
                cv2.circle(refMapVis, (int(pt[0]), int(pt[1])), 2, (255 * (idx / 10),
                                                                    255 * (idx / 10),
                                                                    255 * (idx / 10)), cv2.FILLED)

            for idx, pt in enumerate(estHistory):
                if idx <= 1:
                    continue
                else:
                    cv2.line(refMapVis, (int(pt[0]), int(pt[1])), (int(estHistory[idx - 1][0]), int(estHistory[idx - 1][1])),
                             (0, 0, 255 * idx / len(estHistory)), 3)

            # Show pose estimation on reference map
            cv2.imshow('Pose estimation', cv2.resize(refMapVis, (1000, 1000)))

            # Show zoomed pose estimation on reference map
            zoomRefMapVis = refMapVis[int(estHistory[-1][1]) - 150:int(estHistory[-1][1]) + 150,
                                      int(estHistory[-1][0]) - 150:int(estHistory[-1][0]) + 150]
            cv2.imshow('Zoomed Pose', cv2.resize(zoomRefMapVis, (1000, 1000), cv2.INTER_NEAREST))

        # Press Q on keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cv2.imwrite('full_pose_est.jpg', refMapVis)
        break

# Release when done
cap.release()

# Close all frames
cv2.destroyAllWindows()