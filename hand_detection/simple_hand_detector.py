import cv2
import mediapipe as mp
import time

from hand_detection_module import HandDetector


def canTakePoint(point1, point2, minimumDistance):
    x1, y1 = point1
    x2, y2 = point2
    return (x1-x2)**2 + (y1-y2)**2 >= minimumDistance**2


def drawLineSegments(points: list[list[int]], imageObject):
    for n in range(1, len(points)):
        x1, y1 = points[n]
        x2, y2 = points[n-1]
        cv2.line(imageObject, (x1, y1), (x2, y2), (0, 0, 254), 15)


def drawCurveSegments(line_segments, imageObject):
    for curveSegment in line_segments:
        drawLineSegments(curveSegment, imageObject)


def main():
    cap = cv2.VideoCapture(0)
    handDetector = HandDetector()
    points = [[]]
    line_segments = points[0]

    lastInserted = None
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        #img = cv2.resize(img, (800, 800))
        handDetector.findHands(img, False)
        handLandmarks = handDetector.findhandPosition(
            img, landMarkNumber=4, shouldDraw=True)
        secondHandLandmarks = []
        if handDetector.multi_hand_landmarks and len(handDetector.multi_hand_landmarks) >= 2:
            secondHandLandmarks = handLandmarks = handDetector.findhandPosition(
                img, landMarkNumber=4, shouldDraw=True, handNumber=1)
        if len(handLandmarks) >= 4:
            if len(secondHandLandmarks) < 4:
                # print(handLandmarks[4])
                if lastInserted == None:
                    lastInserted = [handLandmarks[4][1], handLandmarks[4][2]]
                    # print(lastInserted)
                else:
                    if canTakePoint(lastInserted, [handLandmarks[4][1], handLandmarks[4][2]], 25):
                        lastInserted = [handLandmarks[4]
                                        [1], handLandmarks[4][2]]
                        points.append(lastInserted)
            else:
                line_segments.append([])
                points = line_segments[-1]
                lastInserted = None
        drawCurveSegments(line_segments, img)
        # print(points)
        # cv2.putText(img, text="abcd", org=(10, 70), color=(255, 0, 255),
        #             fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=3)
        cv2.imshow("image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
