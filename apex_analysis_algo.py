import cv2
import numpy as np
import time

import parameters as p
from kalman_filter import KalmanFilter
from cv.ego_pose_estimator import PoseEstimator


def updateTimer(timer, update):
    timer = timer * 0.8 + update * 0.2

    return timer


class ApexAnalysisAlgo:
    def __init__(self, refMapPath):
        # Paths
        self.refMap = cv2.imread('data/map/olymp_map.png')
        self.refMapVis = None

        # Sub-algorithms
        self.poseEstimator = PoseEstimator(self.refMap)
        self.kalmanFilter = KalmanFilter(p.KF_TIME_STEP, p.KF_X_ACCEL, p.KF_Y_ACCEL, p.KF_PROC_NOISE,
                                         p.KF_X_STD_MEAS, p.KF_X_STD_MEAS)

        # Counters
        self.poseResetCounter = 0
        self.algoCycleCounter = 0

        # Ego pose estimation
        self.measHistory = np.array([])
        self.estHistory = np.array([])

        # Timers
        self.avPoseEstRunTime = 0
        self.avKalFilRunTime = 0
        self.avVisuRunTime = 0

    def run(self, frame):
        videoFPS = 60

        # Display frame
        cv2.imshow('Current Frame', frame)

        if self.algoCycleCounter % (videoFPS / p.KF_TIME_STEP) == 0:
            # Run analysis algo only once every Kalman Filter time step

            # Initialise visualisation
            self.refMapVis = self.refMap.copy()

            # Run Pose Estimator to obtain a measurement on the ego pose
            startPoseEstimator = time.time()
            newMeas = self.poseEstimator.run(frame)
            endPoseEstimator = time.time()
            self.avPoseEstRunTime = updateTimer(self.avPoseEstRunTime, endPoseEstimator - startPoseEstimator)

            # Run Kalman Filter to obtain filtered pose estimate
            startKalmanFilter = time.time()
            self.runKalmanFilter(newMeas)
            endKalmanFilter = time.time()
            self.avKalFilRunTime = updateTimer(self.avKalFilRunTime, endKalmanFilter - startKalmanFilter)

            # Draw meas and estimated pose history
            startVisu = time.time()
            self.runVisu()
            endVisu = time.time()
            self.avVisuRunTime = updateTimer(self.avVisuRunTime, endVisu - startVisu)

            # Print timers
            self.printTimers()

        self.algoCycleCounter += 1

    def runKalmanFilter(self, newMeas):
        distFromEstPose = 0
        if len(self.measHistory) == 0:
            # Initialise KF state if it hasn't been initialised yet
            self.measHistory = np.array([newMeas])
            self.kalmanFilter.initState(int(self.measHistory[-1][0]), int(self.measHistory[-1][1]))
        else:
            distFromEstPose = (((newMeas[0] - self.estHistory[-1][0]) ** 2)
                               + ((newMeas[1] - self.estHistory[-1][1]) ** 2)) ** 0.5
            self.measHistory = np.append(self.measHistory, [newMeas], 0)

        if distFromEstPose < p.KF_NEW_MEAS_MAX_DIST:
            self.poseResetCounter = 0

            (x, y) = self.kalmanFilter.predict()

            (x1, y1) = self.kalmanFilter.update([self.measHistory[-1][0], self.measHistory[-1][1]])

            if len(self.estHistory) == 0:
                self.estHistory = np.array([[np.asarray(x1)[0][0], np.asarray(y1)[0][1]]])
            else:
                self.estHistory = np.append(self.estHistory, [np.array([np.asarray(x1)[0][0],
                                                                        np.asarray(y1)[0][1]])], 0)
            # Distance of measurement from ego pose is within bounds
            return True
        else:
            self.poseResetCounter = self.poseResetCounter + 1

            if self.poseResetCounter > 5:
                print('Measurement is too far from estimation, and will be ignored. Reset counter = '
                      + str(self.poseResetCounter))

            if self.poseResetCounter > 10:
                # KF = KalmanFilter(1, 0, 0, 1, 5, 5)
                # KF.initState(int(measHistory[-1][0]), int(measHistory[-1][1]))
                # estHistory = np.append(estHistory, [np.array([int(measHistory[-1][0]),
                #                                               int(measHistory[-1][1])])], 0)
                # print('Resetting KF state')
                # resetCounter = 0
                # Draw meas and estimated pose history
                for idx, pt in enumerate(np.flip(np.flip(self.measHistory, 0)[:10], 0)):
                    cv2.circle(self.refMapVis, (int(pt[0]), int(pt[1])), 5, (255 * (idx / 10),
                                                                        255 * (idx / 10),
                                                                        0), cv2.FILLED)

                for idx, pt in enumerate(self.estHistory):
                    if idx <= 1:
                        continue
                    else:
                        cv2.line(self.refMapVis, (int(pt[0]), int(pt[1])),
                                 (int(self.estHistory[idx - 1][0]), int(self.estHistory[idx - 1][1])),
                                 (0, 0, 255 * idx / len(self.estHistory)), 3)

                cv2.imwrite('pose_est_lost_track.jpg', self.refMapVis)

            # New measurement is too far from current pose estimation
            return False

    def runVisu(self):
        for idx, pt in enumerate(np.flip(np.flip(self.measHistory, 0)[:10], 0)):
            cv2.circle(self.refMapVis, (int(pt[0]), int(pt[1])), 5, (255 * (idx / 10),
                                                                255 * (idx / 10),
                                                                0), cv2.FILLED)

        for idx, pt in enumerate(self.estHistory):
            if idx <= 1:
                continue
            else:
                cv2.line(self.refMapVis, (int(pt[0]), int(pt[1])),
                         (int(self.estHistory[idx - 1][0]), int(self.estHistory[idx - 1][1])),
                         (0, 0, 255 * idx / len(self.estHistory)), 3)

        # Show zoomed pose estimation on reference map
        zoomRefMapVis = self.refMapVis[int(self.estHistory[-1][1]) - 150:int(self.estHistory[-1][1]) + 150,
                        int(self.estHistory[-1][0]) - 150:int(self.estHistory[-1][0]) + 150]
        # if estHistory[-1][0] == -1:
        #     zoomRefMapVis = refMapVis

        cv2.imshow('Zoomed Pose', cv2.resize(zoomRefMapVis, (1000, 1000), cv2.INTER_NEAREST))
        # cv2.imshow('RefMapVis', cv2.resize(refMapVis, (1000, 1000)))

    def printTimers(self):
        str_EPE = "{:.2f}".format(self.avPoseEstRunTime)
        str_KF = "{:.2f}".format(self.avKalFilRunTime)
        str_Vis = "{:.2f}".format(self.avVisuRunTime)

        print('EPE: ' + str_EPE + ' KF: ' + str_KF + ' Vis: ' + str_Vis)


if __name__ == '__main__':
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
