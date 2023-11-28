"""
The file is without socket.
main.py will use the action detection function in this file

"""

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import mediapipe as mp
import time
import math

# from signal import signal, SIGPIPE, SIG_DFL
# signal(SIGPIPE, SIG_DFL)

import socket
import json

# Create a socket server
# client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# client_socket.connect(('172.20.10.3', 12345))  # Bind to a specific IP and port


mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_landmarks(image, results):
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS) # Draw face connections
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
    )  # Draw pose connections
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )  # Draw left hand connections
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )  # Draw right hand connections


def extract_keypoints(results, img):
    pose = (
        np.array(
            [
                [res.x, res.y, res.z, res.visibility]
                for res in results.pose_landmarks.landmark
            ]
        ).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )
    # face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = (
        np.array(
            [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
        ).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    rh = (
        np.array(
            [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )
    pose_positions = []
    left_positions = []
    right_positions = []
    if results.pose_landmarks:
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * 1000), int(lm.y * 1000)
            pose_positions.append([id, cx, cy])
    else:
        for i in range(33):
            pose_positions.append([i, 0, 0])
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            left_positions.append([lm.x, lm.y])
    else:
        for i in range(21):
            left_positions.append([0, 0])
    if results.right_hand_landmarks:
        for id, lm in enumerate(results.right_hand_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            right_positions.append([lm.x, lm.y])
    else:
        for i in range(21):
            right_positions.append([0, 0])
    key = np.concatenate([pose, lh, rh])
    return key, pose_positions, left_positions, right_positions


from math import hypot


def distance(A, B):
    length = hypot(B[1] - A[1], B[2] - A[2])
    return int(length)


def angle(A, B):
    delta_x = B[1] - A[1]
    delta_y = B[2] - A[2]

    angle_radians = math.atan2(delta_y, delta_x)
    angle_degrees = math.degrees(angle_radians)

    angle_degrees = (angle_degrees + 360) % 360

    return angle_degrees


# # 4. Setup Folders for Collection


# Actions that we try to detect
actions = np.array(["none", "left", "right", "both"])


from tensorflow.keras.models import load_model

model = load_model("model515.h5", compile=False)
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

w, h = 540, 320

colors = [
    (245, 117, 16),
    (117, 245, 16),
    (16, 117, 245),
    (0, 0, 0),
    (100, 100, 100),
    (150, 156, 217),
]


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(
            output_frame,
            (10, 100 + num * 60),
            (int(prob * 100), 135 + num * 60),
            colors[num],
            -1,
        )
        cv2.putText(
            output_frame,
            actions[num],
            (10, 130 + num * 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output_frame


sequence = []


threshold = 0.99
start = 0

# stream_url = 'http://172.20.10.3:8001/?action=stream'
# cap = cv2.VideoCapture(0)
# Set mediapipe model


def perform_action_detection(frame):
    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        # while cap.isOpened():
        # Read feed
        # ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        # print(results)

        # Draw landmarks
        draw_landmarks(image, results)

        # 2. Prediction logic
        keypoints, pose, left, right = extract_keypoints(results, image)
        # print("left len: ")
        # print(len(left))

        timestamp1 = time.time()

        res = [0] * 5
        ges = 0

        if pose[11][2] > pose[13][2] and pose[12][2] > pose[14][2]:
            res[3] = 1
        # left
        elif angle(pose[11], pose[13]) > 310 or angle(pose[11], pose[13]) < 50:
            res[1] = 1
            pred = model.predict(np.expand_dims(left, axis=0))[0]
            ges = np.argmax(pred).tolist()
            image = prob_viz(
                pred,
                ["no_gesture", "one", "two", "three", "like", "dislike"],
                image,
                colors,
            )
            print(ges, ": ", pred[ges])
            if pred[ges] < threshold:
                ges = 0
        # right
        elif 250 > angle(pose[12], pose[14]) > 130:
            res[2] = 1
            pred = model.predict(np.expand_dims(right, axis=0))[0]
            ges = np.argmax(pred).tolist()
            image = prob_viz(
                pred,
                ["no_gesture", "one", "two", "three", "like", "dislike"],
                image,
                colors,
            )
            print(ges, ": ", pred[ges])
            if pred[ges] < threshold:
                ges = 0
        else:
            res[0] = 1

        # Data to send
        data = {
            "ges": ges,
            "distance": distance(pose[11], pose[12]),
            "mid": (pose[11][1] + pose[12][1]) / 2,
        }

        # # Send the data as a JSON string
        # data_str = json.dumps(data)
        # client_socket.send(data_str.encode())
        # strx = str((pose[11][1] + pose[12][1]) / 2 - 500)
        # if distance > 200  distance < 250:
        #     strr = ""
        # elif distance > 250:
        #     strr = "too close"
        # elif distance < 200 and distance != 0:
        #     strr = "too far"

        # cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        # cv2.putText(
        #     image,
        #     strx + strr,
        #     (3, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1,
        #     (255, 255, 255),
        #     2,
        #     cv2.LINE_AA,
        # )

        # write file
        # pp = 'none'
        # if pose[11][2] > pose[13][2]:
        #     pp ='left'
        # with open('position.txt', 'w') as file:
        #     file.write(f'{pp}, {distance(pose[11], pose[12])},{pose[11]}, {pose[12]}')
        # Show to screen
        cv2.imshow("OpenCV Feed", image)
        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
        return data

    # client_socket.close()


cv2.destroyAllWindows()
