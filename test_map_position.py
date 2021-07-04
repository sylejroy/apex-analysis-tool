import cv2
from cv.map_pos_finder import findMapPoseSIFT


# Reference map path
we_map_path = 'data/map/we_map.png'
test1_path = 'data/screenshots/we/frag_east.PNG'
test2_path = 'data/screenshots/we/harvester.PNG'
test3_path = 'data/screenshots/we/staging.PNG'

# Read images
we_map = cv2.imread(we_map_path)
test1 = cv2.imread(test1_path)
test2 = cv2.imread(test2_path)
test3 = cv2.imread(test3_path)

# Find map pose
#pos = findMapPose(test1, we_map)

findMapPoseSIFT(test1, we_map)
findMapPoseSIFT(test2, we_map)
findMapPoseSIFT(test3, we_map)

