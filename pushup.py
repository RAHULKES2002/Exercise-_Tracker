import cv2
import mediapipe as mp
import numpy as np
import os
import time

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.degrees(radians)
    
    if angle < 0:
        angle += 360
    return angle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

counter = 0
stage = None
output_filename = "output.avi"

def find_pose_landmarks(image, results):
    """Find pose landmarks in the image and draw them."""
    lm_list = []
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = image.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list.append([id, cx, cy])
    return lm_list

def count_push_ups(lm_list):
    """Count push-ups based on shoulder movement."""
    global counter, stage
    if len(lm_list) != 0:
        if lm_list[12][2] >= lm_list[14][2] and lm_list[13][2] >= lm_list[14][2]:
            stage = "down"
        if (lm_list[12][2] <= lm_list[14][2] and lm_list[13][2] <= lm_list[14][2]) and stage == "down":
            stage = "up"
            counter += 1
            print(counter)
            os.system("echo '" + str(counter) + "' | festival --tts")

def main():
    cap = cv2.VideoCapture(0)
    create = None
    try:
        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Failed to read from camera.")
                    break

                image = cv2.resize(image, (1080, 720))
                image_rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                image_rgb.flags.writeable = True
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                lm_list = find_pose_landmarks(image_bgr, results)
                count_push_ups(lm_list)

                text = f"Push Ups: {counter}"
                cv2.putText(image_bgr, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow('MediaPipe Pose', image_bgr)

                if create is None:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    create = cv2.VideoWriter(output_filename, fourcc, 30, (image_bgr.shape[1], image_bgr.shape[0]), True)
                create.write(image_bgr)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
        if create is not None:
            create.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
