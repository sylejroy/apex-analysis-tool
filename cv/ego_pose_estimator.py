import cv2
import numpy as np
import time

import parameters as p


class PoseEstimator:
    def __init__(self, referenceMap):
        self.refMap = cv2.cvtColor(referenceMap, cv2.COLOR_BGR2GRAY)
        self.miniMap = None

        # Preprocess reference map features
        self.feature = cv2.SIFT_create()
        self.refKP, self.refDes = self.feature.detectAndCompute(self.refMap, None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        self.mmKP = None
        self.mmDes = None
        self.matches = None
        self.maskedMatches = None

        self.refMatchedKPList = None
        self.mmMatchedKPList = None

        self.crossDistsRef = []
        self.crossDistsMM = []
        self.estRatio = []
        self.ratioIndex = 0

    def preprocess(self, inputFrame):
        inputFrame = cv2.cvtColor(inputFrame, cv2.COLOR_BGR2GRAY)
        self.miniMap = inputFrame[p.MM_TOP_LEFT_Y_PXL:p.MM_TOP_LEFT_Y_PXL + p.MM_HEIGHT_PXL,
                                  p.MM_TOP_LEFT_X_PXL:p.MM_TOP_LEFT_X_PXL + p.MM_WIDTH_PXL]

        # Reset masked matches
        self.maskedMatches = []
        self.refMatchedKPList = []
        self.mmMatchedKPList = []

        # if p.EPE_DBG:
        #     cv2.imshow('miniMap', self.miniMap)
        #     cv2.imshow('refMap', cv2.resize(self.refMap, (1000, 1000)))

    def matchMinimapToReference(self):
        # Find features on minimap
        self.mmKP, self.mmDes = self.feature.detectAndCompute(self.miniMap, None)
        # Match minimap descriptors with reference map descriptors
        self.matches = self.flann.knnMatch(self.refDes, self.mmDes, k=2)
        # Create mask to only draw good matches
        matchesMask = [[0, 0] for i in range(len(self.matches))]
        # Ratio test to only select good matches
        for i, (m, n) in enumerate(self.matches):
            if m.distance < 0.25 * n.distance:
                matchesMask[i] = [1, 0]
                self.maskedMatches.append(m)

        if p.EPE_DBG:
            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               matchesMask=matchesMask,
                               flags=cv2.DrawMatchesFlags_DEFAULT)
            drawMatches = cv2.drawMatchesKnn(self.refMap, self.refKP, self.miniMap, self.mmKP, self.matches,
                                             None, **draw_params)
            cv2.imshow('matches', cv2.resize(drawMatches, (1000, 1000)))
            cv2.waitKey(1)

    def estEgoPoseFromMatches(self):
        for match in self.maskedMatches:
            refIdx = match.queryIdx
            mmIdx = match.trainIdx
            (x_ref, y_ref) = self.refKP[refIdx].pt
            (x_mm, y_mm) = self.mmKP[mmIdx].pt
            self.refMatchedKPList.append((x_ref, y_ref))
            self.mmMatchedKPList.append((x_mm - p.MM_WIDTH_PXL / 2, y_mm - p.MM_HEIGHT_PXL / 2))

        posEstList = []

        for idx, mmPt in enumerate(self.mmMatchedKPList):
            posEstList.append(np.array(self.refMatchedKPList[idx]) - (np.array(mmPt) * p.MM_TO_REF_SCALE_RATIO))

        pos = sum(posEstList) / len(posEstList)

        print(pos)

        return pos

    def computeScaleRatio(self):
        # Not quite working - occasional division by zero
        for idxRef, ptRef in enumerate(self.refMatchedKPList):
            for idxTest, ptTest in enumerate(self.refMatchedKPList):
                if idxRef != idxTest:
                    self.crossDistsRef.append((ptRef[0] - ptTest[0]) ** 2 + (ptRef[1] - ptTest[1]) ** 2)

        for idxTest, ptTest in enumerate(self.mmMatchedKPList):
            for idxRef, ptRef in enumerate(self.mmMatchedKPList):
                if idxRef != idxTest:
                    self.crossDistsMM.append((ptRef[0] - ptTest[0]) ** 2 + (ptRef[1] - ptTest[1]) ** 2)
                    self.estRatio.append(self.crossDistsRef[self.ratioIndex] / self.crossDistsMM[self.ratioIndex])
                    self.ratioIndex = self.ratioIndex + 1

        # scale_ratio = max(set(self.estRatio), key=self.estRatio.count)
        scale_ratio = sum(self.estRatio) / len(self.estRatio)

        print(scale_ratio)


if __name__ == '__main__':
    PE = PoseEstimator(cv2.imread('../data/map/we_map.png'))
    PE.preprocess(cv2.imread('../data/screenshots/we/staging.PNG'))
    PE.matchMinimapToReference()
    PE.estEgoPoseFromMatches()

    PE.preprocess(cv2.imread('../data/screenshots/we/harvester.PNG'))
    PE.matchMinimapToReference()
    PE.estEgoPoseFromMatches()

    PE.preprocess(cv2.imread('../data/screenshots/we/frag_east.PNG'))
    PE.matchMinimapToReference()
    PE.estEgoPoseFromMatches()

    cv2.waitKey()
