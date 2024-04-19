import cv2
import mediapipe as mp
import numpy as np
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

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

counter_left = 0 
counter_right = 0 
cooldown_time = 2  
last_rep_time = time.time()  
left_curling = False
right_curling = False

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
      
        image = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        results = pose.process(image)
    
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark
            
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            

            angle_left = calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            angle_right = calculate_angle(right_shoulder, right_elbow, right_wrist)

            if angle_left < 90:
                if not left_curling:
                    left_curling = True
            else:
                if left_curling:
                    current_time = time.time()
                    if current_time - last_rep_time > cooldown_time:
                        counter_left += 1
                        last_rep_time = current_time
                        print(f"Left Counter: {counter_left}")
                    left_curling = False
            

            if angle_right < 90:
                if not right_curling:
                    right_curling = True
            else:
                if right_curling:
                    current_time = time.time()
                    if current_time - last_rep_time > cooldown_time:
                        counter_right += 1
                        last_rep_time = current_time
                        print(f"Right Counter: {counter_right}")
                    right_curling = False
                       
        except:
            pass

        cv2.putText(image, f"Left Hand Count: {counter_left}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"Right Hand Count: {counter_right}", 
                    (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        out.write(image)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

out.release()
cap.release()
cv2.destroyAllWindows()