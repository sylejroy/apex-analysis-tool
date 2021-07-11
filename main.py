import cv2
import numpy as np
import time

from apex_analysis_algo import ApexAnalysisAlgo
import parameters as p


algo = ApexAnalysisAlgo(p.REF_MAP_PATH)

# Read input video
cap = cv2.VideoCapture(p.VID_PATH)

# Check if vid is opened successfully
if not cap.isOpened():
    print("Unable to open video: " + p.VID_PATH)

# Loop over all frames in video
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Run algo
    if ret:
        algo.run(frame)

    # Press Q on keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release when doneq
cap.release()

# Close all frames
cv2.destroyAllWindows()