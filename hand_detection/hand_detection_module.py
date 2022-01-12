import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2, model_complexity=1, min_detection_confidence=.5, min_tracking_confidence=.5) -> None:
        self.static_image_mode = static_image_mode,
        self.max_num_hands = max_num_hands,
        self.model_complexity = model_complexity,
        self.min_detection_confidence = min_detection_confidence,
        self.min_tracking_confidence = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.hands = self.mpHands.Hands(
            static_image_mode, max_num_hands, model_complexity, min_detection_confidence, min_tracking_confidence)

    def findHands(self, img, shouldDraw=False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self. hands.process(imgRGB)

        # print(result.multi_hand_landmarks)

        self.multi_hand_landmarks = result.multi_hand_landmarks
        if self.multi_hand_landmarks:
            for hand in self.multi_hand_landmarks:

                if shouldDraw:
                    self.mpDraw.draw_landmarks(
                        img, hand, self.mpHands.HAND_CONNECTIONS)

    def findhandPosition(self, img, handNumber=0, shouldDraw=False):
        landmarkList = []
        hgt, wdt, c = img.shape
        if self.multi_hand_landmarks:
            # print(self.multi_hand_landmarks)
            if len(self.multi_hand_landmarks) >= handNumber:
                for id, lm in enumerate(self.multi_hand_landmarks[handNumber].landmark):
                    cx, cy = int(lm.x*wdt), int(lm.y*hgt)
                    print(cx, cy)
                    landmarkList.append([id, cx, cy])
                    if shouldDraw:
                        cv2.circle(img, (cx, cy), 15,
                                   (255, 0, 255), cv2.FILLED)
        return landmarkList


def main():
    cap = cv2.VideoCapture(0)
    handDetector = HandDetector()
    while True:
        success, img = cap.read()
        handDetector.findHands(img, True)
        handDetector.findhandPosition(img)
        # cv2.putText(img, text="abcd", org=(10, 70), color=(255, 0, 255),
        #             fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=3)
        cv2.imshow("image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
