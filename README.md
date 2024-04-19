# Exercise Tracker
Certainly! Here's a simple README file for your GitHub repository:

---

# Push-up Counter using MediaPipe Pose

This project utilizes [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose) to count push-ups in real-time using a webcam. It tracks the movement of the shoulders to detect push-up repetitions accurately.

## Features
- Counts push-ups based on shoulder movement.
- Provides real-time feedback on the number of push-ups completed.
- Saves a video recording of the push-up session.

## Requirements
- Python 3.x
- OpenCV (cv2)
- MediaPipe
- NumPy

## Installation
1. Clone this repository:

    ```
    git clone https://github.com/yourusername/push-up-counter.git
    ```

2. Install the required dependencies:

    ```
    pip install opencv-python mediapipe numpy
    ```

## Usage
1. Run the `pushup_counter.py` script:

    ```
    python pushup_counter.py
    ```

2. Follow the instructions on the terminal window.
3. Perform push-ups in front of your webcam.
4. See the real-time count of push-ups on the screen.

## Customization
- You can adjust the `cooldown_time` variable in the code to change the cooldown period between each push-up repetition detection.
- Modify the `output_filename` variable to change the name of the output video file.

## Credits
This project was inspired by [MediaPipe](https://google.github.io/mediapipe/), an open-source framework developed by Google.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to customize it further according to your preferences! Let me know if you need any more help.
