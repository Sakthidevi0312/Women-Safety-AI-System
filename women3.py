import cv2
import datetime
import socket
import time
import mediapipe as mp
import numpy as np
import pickle
import paho.mqtt.client as mqtt
import json
import mysql.connector
from twilio.rest import Client


# ---------- Twilio Configuration ----------
account_sid = 'YOUR SID'
auth_token = 'YOUR AUTH TOKEN'
twilio_client = Client(account_sid, auth_token)
twilio_phone_number = 'TWILIO NUMBER'  # Your Twilio number
emergency_contacts = ['YOUR CONTACT NUMBER']

# ---------- MQTT Configuration ----------
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC = "women_safety/alert"

# ---------- Load Face, Gender & Age Models ----------
faceProto = r"D:\detection\opencv_face_detector (1).pbtxt"
faceModel = r"D:\detection\opencv_face_detector_uint8.pb"
ageProto = r"D:\detection\age_deploy.prototxt"
ageModel = r"D:\detection\age_net.caffemodel"
genderProto = r"D:\detection\gender_deploy (1).prototxt"
genderModel = r"D:\detection\gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# ---------- Load Gesture Recognition Model ----------
try:
    with open('gesture_model.pkl', 'rb') as f:
        gesture_model = pickle.load(f)
    print("✅ Gesture model loaded.")
except:
    gesture_model = None
    print("⚠ Gesture model not found. Using dummy classifier.")

# ---------- MediaPipe Hands ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)

# ---------- MQTT Client Setup ----------
mqtt_client = mqtt.Client()

def connect_mqtt():
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        print("✅ Connected to MQTT Broker!")
    except Exception as e:
        print(f"❌ Failed to connect to MQTT Broker: {e}")

def send_alert(message):
    try:
        payload = json.dumps({"message": message})
        mqtt_client.publish(MQTT_TOPIC, payload)
        print(f"🚀 MQTT alert sent: {message}")

        for number in emergency_contacts:
            sms = twilio_client.messages.create(
                from_=twilio_phone_number,
                body=message,
                to=number
            )
            print(f"📩 SMS sent to {number}, SID: {sms.sid}")
    except Exception as e:
        print(f"❌ Error sending alert: {e}")

# ---------- MySQL Database Setup ----------
def connect_db():
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='womendevice',
            port=3306
        )
        print("✅ Connected to MySQL Database!")

        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS safety_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp VARCHAR(50),
                network_status VARCHAR(10),
                persons_detected INT,
                male_count INT,
                female_count INT,
                alert_message VARCHAR(255),
                alone_night BOOLEAN,
                surrounded_by_men BOOLEAN,
                distress_gesture BOOLEAN,
                danger_area BOOLEAN  
            )
        ''')
        conn.commit()
        return conn
    except mysql.connector.Error as err:
        print(f"❌ Database error: {err}")
        return None

def log_to_db(conn, timestamp, network_status, persons_detected, male_count, female_count,
              alert_message=None, alone_night=False, surrounded_by_men=False, distress_gesture=False, danger_area=False):
    try:
        cursor = conn.cursor()
        sql = '''
            INSERT INTO safety_logs (timestamp, network_status, persons_detected, male_count, female_count,
                                     alert_message, alone_night, surrounded_by_men, distress_gesture, danger_area)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        '''
        val = (timestamp, network_status, persons_detected, male_count, female_count,
               alert_message, alone_night, surrounded_by_men, distress_gesture, danger_area)
        cursor.execute(sql, val)
        conn.commit()
        print("📝 Data logged to MySQL Database.")
    except mysql.connector.Error as err:
        print(f"❌ Failed to log data: {err}")

# ---------- Utility Functions ----------
def check_network():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=1)
        return "Online"
    except OSError:
        return "Offline"

def getFaceBox(net, frame, conf_threshold=0.7):
    frameHeight, frameWidth = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
    return frame.copy(), bboxes

def extract_hand_keypoints(hand_landmarks):
    return [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]

def dummy_predict(keypoints):
    return 'distress' if len(keypoints) == 63 else 'safe'

# ---------- GPS Danger Hotspot Checking ----------
current_latitude = 8.78244
current_longitude = 77.61206
danger_lat = 9.9563
danger_long = 78.0701

def check_hotspot_area(latitude, longitude):
    if (abs(latitude - danger_lat) < 0.1) and (abs(longitude - danger_long) < 0.1):
        return True
    return False

# ---------- Main Program ----------
def main():
    connect_mqtt()
    db_conn = connect_db()

    if db_conn is None:
        print("🚫 Exiting due to database connection failure.")
        return

    cap = cv2.VideoCapture(0)
    print("📹 Starting camera...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        now = datetime.datetime.now()
        dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
        hour_now = now.hour
        network_status = check_network()

        frameFace, bboxes = getFaceBox(faceNet, frame)
        male_count, female_count = 0, 0
        alone_night = False
        surrounded_by_men = False
        distress_detected = False

        print(f"\n📅 {dt_string}")
        print(f"🌐 Network: {network_status}")
        print(f"👥 Persons Detected: {len(bboxes)}")

        danger_area = check_hotspot_area(current_latitude, current_longitude)
        if danger_area:
            print("⚠ Danger area spotted!")
            send_alert("⚠ Danger area detected! Stay Alert!")
        else:
            print("✅ No danger spotted.")

        for bbox in bboxes:
            face = frame[max(0, bbox[1]):min(bbox[3], frame.shape[0]-1),
                         max(0, bbox[0]):min(bbox[2], frame.shape[1]-1)]
            if face.shape[0] == 0 or face.shape[1] == 0:
                continue

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                         [78.426, 87.769, 114.896], swapRB=False)
            genderNet.setInput(blob)
            gender = GENDER_LIST[genderNet.forward()[0].argmax()]
            ageNet.setInput(blob)
            age = AGE_LIST[ageNet.forward()[0].argmax()]

            if gender == 'Male':
                male_count += 1
            else:
                female_count += 1

            label = f"{gender}, {age}"
            cv2.rectangle(frameFace, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frameFace, label, (bbox[0], bbox[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Women Safety Detections
        if (hour_now >= 20 or hour_now < 6):
            if female_count >= 1 and male_count >= 3:
                alone_night = True
                cv2.putText(frameFace, "⚠️ AloneAtNight (Female + 3+ Males)", (10, 175),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print("⚠️ AloneAtNight: Female with 3+ Males")
                send_alert("⚠ Alone at Night: Female with 3+ Males!")
            elif female_count == 1 and male_count == 0:
                alone_night = True
                cv2.putText(frameFace, "⚠️ AloneAtNight (Female Alone)", (10, 175),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print("⚠️ AloneAtNight: Female Alone")
                send_alert("⚠ Alone at Night: Female Alone!")

        if female_count == 1 and male_count >= 4:
            surrounded_by_men = True
            cv2.putText(frameFace, "🚨 SurroundedByMen (1 Female + 4+ Males)", (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print("🚨 SurroundedByMen: 1 Female surrounded by 4+ Males")
            send_alert("🚨 ALERT: 1 Female surrounded by 4+ Males!")

        # Hand Gesture Detection
        hand_result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if hand_result.multi_hand_landmarks:
            for hand_landmarks in hand_result.multi_hand_landmarks:
                hand_keypoints = extract_hand_keypoints(hand_landmarks)
                gesture = dummy_predict(hand_keypoints)

                if gesture == "distress":
                    distress_detected = True
                    print("🚨 Distress gesture detected!")
                    send_alert("🚨 Distress Gesture Detected!")

        # Log to DB
        log_to_db(db_conn, dt_string, network_status, len(bboxes), male_count, female_count,
                  alert_message=None, alone_night=alone_night,
                  surrounded_by_men=surrounded_by_men,
                  distress_gesture=distress_detected,
                  danger_area=danger_area)

        cv2.imshow("Women Safety Surveillance", frameFace)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
