import cv2
import mediapipe as mp
import numpy as np
import os

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle 

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


cap = cv2.VideoCapture(0)

counter_left = 0 
counter_right = 0 
counter_squats = 0
stage_left = None
stage_right = None
squat_flag = False
counter = 0
stage = None
create = None
opname = "output.avi"

def findPosition(image, draw=True):
    lmList = []
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = image.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy])
           
    return lmList

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
            
            
            cv2.putText(image, f"Left Angle: {angle_left:.2f}", 
                           (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
                                )
            
            cv2.putText(image, f"Right Angle: {angle_right:.2f}", 
                           (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
                                )
            
            
            if angle_left > 160:
                stage_left = "Down"
            if angle_left < 30 and stage_left =='Down':
                stage_left="Up"
                counter_left += 1
                print(f"Left Counter: {counter_left}")
            
           
            if angle_right > 160:
                stage_right = "Down"
            if angle_right < 30 and stage_right =='Down':
                stage_right="Up"
                counter_right += 1
                print(f"Right Counter: {counter_right}")
                       
        except:
            pass

       
        try:
            
            left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y
            right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y
            nose_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y

            if nose_y < left_hip and nose_y < right_hip:
                if not squat_flag:
                    squat_flag = True
            else:
                if squat_flag:
                    counter_squats += 1
                    print(f"Squat Counter: {counter_squats}")
                    squat_flag = False

        except:
            pass
        
       
        cv2.putText(image, f"Left Hand Count: {counter_left}", 
                    (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"Right Hand Count: {counter_right}", 
                    (10, 190), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        #cv2.putText(image, f"Squat Count: {counter_squats}", 
         #           (10, 230), 
          #          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

       
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2) 
                                 )               
        
       
        lmList = findPosition(image, draw=True)
        if len(lmList) != 0:
            cv2.circle(image, (lmList[12][1], lmList[12][2]), 20, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, (lmList[11][1], lmList[11][2]), 20, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, (lmList[12][1], lmList[12][2]), 20, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, (lmList[11][1], lmList[11][2]), 20, (0, 0, 255), cv2.FILLED)
            if (lmList[12][2] and lmList[11][2] >= lmList[14][2] and lmList[13][2]):
                cv2.circle(image, (lmList[12][1], lmList[12][2]), 20, (0, 255, 0), cv2.FILLED)
                cv2.circle(image, (lmList[11][1], lmList[11][2]), 20, (0, 255, 0), cv2.FILLED)
                stage = "down"
            if (lmList[12][2] and lmList[11][2] <= lmList[14][2] and lmList[13][2]) and stage == "down":
                stage = "up"
                counter += 1
                counter2 = str(int(counter))
                print(counter)
                os.system("echo '" + counter2 + "' | festival --tts")
        text = "{}:{}".format("Push Ups", counter)
        cv2.putText(image, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        
        cv2.imshow('Mediapipe Feed', image)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    
        if create is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            create = cv2.VideoWriter(opname, fourcc, 30, (image.shape[1], image.shape[0]), True)
        create.write(image)

cap.release()
cv2.destroyAllWindows()