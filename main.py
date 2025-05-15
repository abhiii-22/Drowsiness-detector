import cv2
import time
import mediapipe as mp
import winsound
import threading
import csv
from datetime import datetime
from math import dist

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Eye landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Constants
EYE_AR_THRESH = 0.22  
CLOSED_FRAMES_REQUIRED = 20  
BLINK_MAX = 10
FONT = cv2.FONT_HERSHEY_SIMPLEX

# State variables
counter = 0
blink_counter = 0
alarm_on = False
beep_thread = None
beep_active = False
sleep_start = None
total_sleep_time = 0
log_file = 'sleep_log.csv'

# Logging file
with open(log_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Status", "Duration_sec"])

def sound_alarm():
    global beep_active
    beep_active = True
    while beep_active:
        winsound.Beep(1000, 800)
        time.sleep(0.5)

def stop_alarm():
    global beep_active
    beep_active = False

def eye_aspect_ratio(eye):
    A = dist(eye[1], eye[5])
    B = dist(eye[2], eye[4])
    C = dist(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def draw_status_box(img, text, color):
    (w, h), _ = cv2.getTextSize(text, FONT, 1.2, 2)
    x, y = 20, 40
    cv2.rectangle(img, (x - 10, y - 35), (x + w + 20, y + 10), color, -1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, 1.2, (255, 255, 255), 2, cv2.LINE_AA)


cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

prev_closed = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE]
            right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE]

            ear_left = eye_aspect_ratio(left_eye)
            ear_right = eye_aspect_ratio(right_eye)
            ear = (ear_left + ear_right) / 2.0

            eye_color = (0, 255, 0) if ear > EYE_AR_THRESH else (0, 0, 255)
            for pt in left_eye + right_eye:
                cv2.circle(frame, pt, 2, eye_color, -1)

            if ear < EYE_AR_THRESH:
                counter += 1
                if BLINK_MIN <= counter <= BLINK_MAX and not prev_closed:
                    blink_counter += 1
                    prev_closed = True

                if counter >= CLOSED_FRAMES_REQUIRED:
                    if not alarm_on:
                        alarm_on = True
                        if beep_thread is None or not beep_thread.is_alive():
                            beep_thread = threading.Thread(target=sound_alarm)
                            beep_thread.daemon = True
                            beep_thread.start()
                        sleep_start = time.time()
                    draw_status_box(frame, "INATTENTIVE", (0, 0, 150))
                else:
                    draw_status_box(frame, "ATTENTIVE", (30, 180, 90))
            else:
                if alarm_on:
                    alarm_on = False
                    stop_alarm()
                    if sleep_start:
                        sleep_duration = round(time.time() - sleep_start, 2)
                        total_sleep_time += sleep_duration
                        with open(log_file, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Sleep", sleep_duration])
                        sleep_start = None
                counter = 0
                prev_closed = False
                draw_status_box(frame, "ATTENTIVE", (30, 180, 90))

            # Overlay metrics
            cv2.putText(frame, f"Blinks: {blink_counter}", (20, h - 60), FONT, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Sleep Time: {int(total_sleep_time)}s", (20, h - 30), FONT, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"EAR: {ear:.2f}", (20, h - 90), FONT, 0.7, (255, 255, 255), 2)

    else:
        draw_status_box(frame, "No person detected", (50, 50, 150))

    cv2.imshow("Drowsiness Detector Pro", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_alarm()
        break

cap.release()
cv2.destroyAllWindows()
