"""
This file is for testing 
1. The action detection is success
2. The follow algo is correct

"""
from Action_Detection_video import perform_action_detection
from djitellopy import Tello
import time
import numpy as np

pError = 0
pid = [0.15, 0.17, 0]


def init():
    global tello
    tello = Tello()
    tello.connect()
    print(tello.get_battery())
    tello.streamon()


def takeoff_to_height():
    tello.takeoff()
    tello.send_rc_control(0, 0, 22, 0)
    time.sleep(5)


def land_and_close():
    tello.streamoff()
    tello.land()
    tello.end()
    print("Land vehicle and close the camera")


def trackFace(center_x, distance, pid, pError):
    fb = 0
    error = center_x - 1000 // 2

    speed = pid[0] * error + pid[1] * (error - pError)

    speed = int(np.clip(speed, -100, 100))

    if distance > 150 and distance < 190:
        fb = 0
    elif distance > 190:
        fb = -20
    elif distance < 160 and distance != 0:
        fb = 20
    if center_x == 0:
        speed = 0
        error = 0

    tello.send_rc_control(0, fb, 0, speed)
    return error


if __name__ == "__main__":
    # init tello, and takeoff to human height
    init()
    takeoff_to_height()
    # Yaw to detect a human
    initial_degree = tello.get_yaw()
    while True:
        frame = tello.get_frame_read().frame
        data = perform_action_detection(frame)
        distance = data["distance"]
        mid = data["mid"]
        current_degree = tello.get_yaw()
        yaw_difference = current_degree - initial_degree
        # If human found, stop yawing
        if distance and mid:
            break
        if yaw_difference >= 360:
            land_and_close()
        # If no human found, continue yawing
        else:
            tello.send_rc_control(0, 0, 0, 30)

    # set initial mode to follow human
    mode = 0
    log = []
    try:
        while True:
            # Get the current frame from the Tello camera
            frame = tello.get_frame_read().frame
            data = perform_action_detection(frame)
            ges = data["ges"]
            distance = data["distance"]
            mid = data["mid"]

            if distance and mid:
                print(distance)
                print(ges)
                print(mid)
                log.append(ges)
                if len(log) > 10:
                    last = log[-11]
                    log = log[-10:]
                    if all(x == log[0] for x in log) and log[0] != last:
                        print(f"gesture detect: {ges}")
                        mode = log[0]
                if mode == 0:
                    print("mode 0")
                    # Your code to follow the human goes here
                    center_x = mid
                    pError = trackFace(center_x, distance, pid, pError)

                    # Check if human has a gesture

                elif log[0] == 1:
                    print("Flight right started")
                    tello.send_rc_control(20, 0, 0, 0)
                    time.sleep(4)
                    print("Flight right complete")
                    mode = 0
                elif log[0] == 2:
                    print("Flight left started")
                    tello.send_rc_control(-20, 0, 0, 0)
                    time.sleep(4)
                    print("Flight left complete")
                    mode = 0
                elif log[0] == 3:
                    # Land the vehicle and close the camera
                    land_and_close()
                    break
            # If no human found, continue yawing
            else:
                print("no object detected")
                tello.send_rc_control(0, 0, 0, 30)
 
    except:
        land_and_close()
