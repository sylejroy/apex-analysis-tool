import cv2
from cv.map_pos_finder import findMapPoseBRISK

# Video name
vid_path = 'data/vids/lifeline 4k.mp4'

# Read video
cap = cv2.VideoCapture(vid_path)

# Check if vid is opened successfully
if (cap.isOpened() == False):
    print("Unable to open video: " + vid_path)

index = 0
poseHistory = []

# Read until video is finished
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    index = index + 1
    if (ret == True):
        # Display frame
        cv2.imshow(vid_path, frame)
        if (index % 10 == 0):
            # Find map pose estimate
            refMap = cv2.imread('data/map/we_map.png')
            poseHistory.append(findMapPoseBRISK(frame, refMap))
            for pt in poseHistory:
                cv2.circle(refMap, (int(pt[0]), int(pt[1])), 1, (0, 0, 255), 1)
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