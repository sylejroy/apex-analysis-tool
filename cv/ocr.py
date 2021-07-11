import cv2
import pytesseract
from pytesseract import Output

class OCR:
    def __init__(self, template):
        self.template = template
        self.topLeftMatch = None
        self.botRightMatch = None

        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
        return

    def matchTemplate(self, frame):
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        w, h = self.template.shape[::-1]

        frame = cv2.threshold(frame, 180, 255, cv2.THRESH_BINARY)
        
        cv2.imshow('frame', frame)
        cv2.imshow('template', self.template)

        method = cv2.TM_SQDIFF_NORMED

        res = cv2.matchTemplate(frame, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        # if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        #     top_left = min_loc
        # else:
        #     top_left = max_loc

        print(min_val)
        print(min_loc)

        self.topLeftMatch = min_loc
        self.botRightMatch = (self.topLeftMatch[0] + w, self.topLeftMatch[1] + h)
        cv2.rectangle(frame, self.topLeftMatch, self.botRightMatch, 0, 2)

        # cv2.circle(frame, min_loc, 30, (0, 0, 255))
        cv2.imshow('matched', frame)

        ocrArea = frame[self.topLeftMatch[1]: self.topLeftMatch[1] + 30, self.botRightMatch[0]: self.botRightMatch[0] + 15]

        cv2.imshow('ocr_area', ocrArea)

        return ocrArea

    def runOCR(self, ocrArea):
        # Adding custom options
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(ocrArea, config=custom_config)

        print(text)

        return text


if __name__ == '__main__':
    template = cv2.imread('../data/templates/kills.PNG', 0)
    frame = cv2.imread('../data/screenshots/we/staging.PNG', 0)

    ocr = OCR(template)

    ocrArea = ocr.matchTemplate(frame)
    text = ocr.runOCR(ocrArea)

    print(text)

    cv2.waitKey()

