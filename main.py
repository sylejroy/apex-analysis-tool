import cv2

# Video name
vid_path = 'data/vids/death.mp4'

# Read video
cap = cv2.VideoCapture(vid_path)

# Check if vid is opened successfully
if (cap.isOpened() == False):
    print("Unable to open video: " + vid_path)

# Read until video is finished
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
            # Display frame
            cv2.imshow(vid_path, frame)

            # Press Q on keyboard to exit
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    else:
        break

# Release when done
cap.release()

# Close all frames
cv2.destroyAllWindows()