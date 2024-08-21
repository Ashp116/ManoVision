import cv2
import mediapipe as mp
import numpy as np

class HandDetector():
    def __init__(self, mode=False, maxHands=2, detection_confidence=0.5, track_confidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detection_confidence,
                                        min_tracking_confidence=self.track_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw=False):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw: self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
        return frame

    def get_hand(self, handNo):
        try:
            return self.results.multi_hand_landmarks[handNo]
        except:
            return None
    def find_position(self, frame, draw=False):
        lmList = {}

        if self.results.multi_hand_landmarks:
            for i, myHand in enumerate(self.results.multi_hand_landmarks):
                label = self.get_label(i)
                if label:
                    lmList[label] = {}
                    for id, lm in enumerate(myHand.landmark):
                        h, w, c = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList[label][id] = np.array([cx, cy, lm.x, lm.y])

                        if draw: cv2.circle(frame, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
        return lmList

    def get_label(self, index):
        output = None
        if self.results.multi_handedness[index]:
            classification = self.results.multi_handedness[index]
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = '{}'.format(label)

            output = text

        # for idx, classification in enumerate(self.results.multi_handedness):
        #     if classification.classification[0].index == index:
        #         # Process results
        #         label = classification.classification[0].label
        #         score = classification.classification[0].score
        #         text = '{}'.format(label)
        #
        #         output = text

        return output
