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

    def run(self, inputFrame):
        self.preprocess(inputFrame)

        # Check for empty descriptors
        if self.mmDes is None or self.refDes is None:
            return [-1, -1]
        else:
            if len(self.mmDes) < 2:
                return [-1, -1]

        self.matchMinimapToReference()
        posEstimation = self.estEgoPoseFromMatches()
        self.computeMatchValidation()
        return posEstimation

    def preprocess(self, inputFrame):
        inputFrame = cv2.cvtColor(inputFrame, cv2.COLOR_BGR2GRAY)
        self.miniMap = inputFrame[p.MM_TOP_LEFT_Y_PXL:p.MM_TOP_LEFT_Y_PXL + p.MM_HEIGHT_PXL,
                                  p.MM_TOP_LEFT_X_PXL:p.MM_TOP_LEFT_X_PXL + p.MM_WIDTH_PXL]

        # Reset masked matches
        self.maskedMatches = []
        self.refMatchedKPList = []
        self.mmMatchedKPList = []

        # Find features on minimap
        self.mmKP, self.mmDes = self.feature.detectAndCompute(self.miniMap, None)

        # if p.EPE_DBG:
        #     cv2.imshow('miniMap', self.miniMap)
        #     cv2.imshow('refMap', cv2.resize(self.refMap, (1000, 1000)))

    def matchMinimapToReference(self):
        # Match minimap descriptors with reference map descriptors
        self.matches = self.flann.knnMatch(self.refDes, self.mmDes, k=2)
        # Create mask to only draw good matches
        matchesMask = [[0, 0] for _ in range(len(self.matches))]
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
        prevXref = 0

        for match in self.maskedMatches:
            refIdx = match.queryIdx
            mmIdx = match.trainIdx
            (x_ref, y_ref) = self.refKP[refIdx].pt
            (x_mm, y_mm) = self.mmKP[mmIdx].pt
            if int(x_ref) != prevXref:
                self.refMatchedKPList.append((x_ref, y_ref))
                # Translate minimap key points to centre of minimap
                self.mmMatchedKPList.append((x_mm - p.MM_WIDTH_PXL / 2, y_mm - p.MM_HEIGHT_PXL / 2))
                prevXref = int(x_ref)

        posEstList = []

        for idx, mmPt in enumerate(self.mmMatchedKPList):
            posEstList.append(np.array(self.refMatchedKPList[idx]) - (np.array(mmPt) * p.MM_TO_REF_SCALE_RATIO))

        if len(posEstList) == 0:
            pos = [-1, -1]
        else:
            pos = sum(posEstList) / len(posEstList)

        return pos

    def computeMatchValidation(self):
        refOutlierIdxList = []
        mmOutlierIdxList = []

        for idxRef, ptRef in enumerate(self.refMatchedKPList):
            currentCrossDists = []
            for idxTst, ptTst in enumerate(self.refMatchedKPList):
                if idxRef != idxTst:
                    self.crossDistsRef.append(((ptRef[0] - ptTst[0]) ** 2 + (ptRef[1] - ptTst[1]) ** 2) ** 0.5)
                    currentCrossDists.append(self.crossDistsRef[-1])
                    if len(currentCrossDists) > 1:
                        if (sum(currentCrossDists) / len(currentCrossDists)) > 300:
                            # Most likely an outlier
                            refOutlierIdxList.append(idxRef)

        for idxRef, ptRef in enumerate(self.mmMatchedKPList):
            currentCrossDists = []
            for idxTst, ptTst in enumerate(self.mmMatchedKPList):
                if idxRef != idxTst:
                    self.crossDistsMM.append(((ptRef[0] - ptTst[0]) ** 2 + (ptRef[1] - ptTst[1]) ** 2) ** 0.5)
                    currentCrossDists.append(self.crossDistsMM[-1])
                    if len(currentCrossDists) != 0:
                        if (sum(currentCrossDists) / len(currentCrossDists)) > 400:
                            # Most likely an outlier
                            mmOutlierIdxList.append(idxRef)

        if len(refOutlierIdxList) != 0:
            for i in refOutlierIdxList.reverse():
                del self.refMatchedKPList[i]

        if len(mmOutlierIdxList) != 0:
            for i in mmOutlierIdxList.reverse():
                del self.mmMatchedKPList[i]


        # # This scale ratio computation is still problematic as it sometimes leads to a division by zero
        #             self.estRatio.append(self.crossDistsRef[self.ratioIndex] / self.crossDistsMM[self.ratioIndex])
        #             self.ratioIndex = self.ratioIndex + 1
        #
        # # scale_ratio = max(set(self.estRatio), key=self.estRatio.count)
        # scale_ratio = sum(self.estRatio) / len(self.estRatio)
        #
        # print(scale_ratio)


if __name__ == '__main__':
    PE = PoseEstimator(cv2.imread('../data/map/we_map.png'))
    #PE.run(cv2.imread('../data/screenshots/we/staging.PNG'))
    PE.run(cv2.imread('../data/screenshots/we/harvester.PNG'))
    #PE.run(cv2.imread('../data/screenshots/we/frag_east.PNG'))

    cv2.waitKey()
