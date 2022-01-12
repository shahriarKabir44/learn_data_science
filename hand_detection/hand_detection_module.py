import cv2
import mediapipe as mp
import time

from mediapipe.python.solutions import hands

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands

mpDraw = mp.solutions.drawing_utils

hands = mpHands.Hands()


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    hgt, wdt, c = img.shape
    # print(result.multi_hand_landmarks)

    multi_hand_landmarks = result.multi_hand_landmarks
    if multi_hand_landmarks:
        for hand in multi_hand_landmarks:
            # for id, lm in enumerate(hand.landmark):
            #     cx, cy = int(lm.x*wdt), int(lm.y*hgt)
            #     if id == 3:
            #         cv2.circle(img, (cx, cy), 20, (255, 0, 255))
            mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)

    cv2.putText(img, text="abcd", org=(10, 70), color=(255, 0, 255),
                fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=3)
    cv2.imshow("image", img)
    cv2.waitKey(1)
