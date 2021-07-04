import cv2
from cv.map_pos_finder import findMapPoseSIFT

# Video name
vid_path = 'data/vids/death.mp4'

# Read video
cap = cv2.VideoCapture(vid_path)

# Check if vid is opened successfully
if (cap.isOpened() == False):
    print("Unable to open video: " + vid_path)

# Read until video is finished
index = 0

while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    index = index + 1
    if (ret == True):
        if (index % 20 == 0):
            # Display frame
            cv2.imshow(vid_path, frame)
            findMapPoseSIFT(frame, cv2.imread('data/map/we_map.png'))
            # Press Q on keyboard to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        break

# Release when done
cap.release()

# Close all frames
cv2.destroyAllWindows()