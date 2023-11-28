# Tello Gesture Control & Human Following

## Introduction

Welcome to the Tello Gesture Control & Human Following project! This repository integrates the tello, OpenCV, and MediaPipe technologies to achieve automatic tracking of specific individuals and gesture-based control for Tello drones.

## Features

- **Automatic Human Following:** Utilize the capabilities of MediaPipe for real-time human detection and tracking, enabling the drone to autonomously follow a specific person.

- **Gesture Control:** Implement gesture recognition using MediaPipe to allow users to interact with the drone through predefined hand gestures, providing an intuitive and hands-free control experience.

## Getting Started

### Prerequisites

- Tello drone
- Python 3

### File explanation

1. Action_Detection_video.py:
   - gesture estimation using mediapipe, the result will be each frame's gesture
   - return the important value for human-following function, included shoulder distance and human center

2. main_test_integrate.py:
   - control the takeoff and land of tello
   - find human automatically
   - follow human
   - use the gesture result to make corresponding flight control

3. model515.h5:
  - model for gesture detection

## Usage

- Ensure that the Tello drone is connected to the computer using wi-fi, and the necessary setup is complete.

- ensure that you have install all package that will use in the file

- Execute the main_test_integrate.py script, which initiates the automatic human following and gesture control functionalities.

- Interact with the drone using predefined hand gestures, and observe its response.

## Contributions

Contributions are welcome! If you have ideas for improvements, bug fixes, or new features, feel free to open an issue or submit a pull request.

## Acknowledgments

- The project builds upon the capabilities of PX4, Raspberry Pi, DroneKit, and MediaPipe. Special thanks to their respective communities.

- Inspiration for this project came from the increasing applications of drones and the desire to enhance their capabilities for interactive and autonomous operations.

Thank you for exploring the Tello Gesture Control & Human Following project! Feel free to reach out for any questions or feedback. Happy flying! 🚁✨
