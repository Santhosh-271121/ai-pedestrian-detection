from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import pyttsx3
import threading
import time

app = Flask(__name__)

# 🔥 MODEL
model = YOLO("yolov8m.pt")

# CAMERA
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)

# LABELS
labels = {
    0: "Person",
    2: "Car",
    3: "Motorbike",
    5: "Bus",
    7: "Truck"
}

# 🎤 VOICE ENGINE (AI ASSISTANT STYLE)
engine = pyttsx3.init()
engine.setProperty('rate', 165)
engine.setProperty('volume', 1)

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)  # change index if needed

def speak(text):
    def run():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run).start()

# SMART VOICE CONTROL
last_alert = ""
last_spoken_time = 0

def smart_voice(alert, suggestion):
    global last_alert, last_spoken_time

    current_time = time.time()

    # speak when alert changes OR every 5 sec
    if alert != last_alert or (current_time - last_spoken_time > 5):

        if "🚨" in alert:
            speak("Warning! " + suggestion)

        elif "⚠️" in alert:
            speak("Attention. " + suggestion)

        else:
            speak("All clear. You are safe.")

        last_alert = alert
        last_spoken_time = current_time

# 🌙 NIGHT
def enhance_night(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l,a,b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# 📏 DISTANCE
def estimate_distance(h, H):
    return round((H/(h+1))*2,1)

alert_data = {}

prev_positions = {}
prev_time = time.time()

def gen_frames():
    global alert_data, prev_positions, prev_time

    while True:
        success, frame = cap.read()
        if not success:
            break

        H, W = frame.shape[:2]

        if frame.mean() < 60:
            frame = enhance_night(frame)

        results = model(frame, conf=0.4)

        objects_list = []
        vehicle_count = 0
        person_count = 0

        closest_distance = 999
        obj_name = "None"
        obj_dist = "--"
        direction = "None"

        current_positions = {}

        if results[0].boxes is not None:

            for i, (box, cls) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.cls)):

                x1,y1,x2,y2 = map(int, box)
                label = int(cls)

                name = labels.get(label, "Object")

                height = y2 - y1
                distance = estimate_distance(height, H)

                cx = (x1 + x2)//2

                current_positions[i] = (cx, y1)

                objects_list.append({
                    "name": name,
                    "distance": distance
                })

                # COUNT
                if label == 0:
                    person_count += 1
                else:
                    vehicle_count += 1

                # CLOSEST
                if distance < closest_distance:
                    closest_distance = distance
                    obj_name = name
                    obj_dist = distance

                    if cx < W//3:
                        direction = "Left"
                    elif cx > 2*W//3:
                        direction = "Right"
                    else:
                        direction = "Center"

                # DRAW
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,f"{name} {distance}m",
                            (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)

        prev_positions = current_positions
        prev_time = time.time()

        total_objects = len(objects_list)
        crowd = total_objects >= 5

        # 🎯 ALERT LOGIC
        if crowd:
            alert = "🚨 CROWD DETECTED"
            suggestion = "Multiple objects ahead. Drive carefully."

        elif closest_distance < 5:
            alert = f"🚨 HIGH RISK - {obj_name}"
            suggestion = f"{obj_name} very close. Stop immediately."

        elif closest_distance < 10:
            alert = f"⚠️ WARNING - {obj_name}"
            suggestion = f"{obj_name} ahead. Reduce speed."

        else:
            alert = "🟢 SAFE"
            suggestion = "Road is clear."

        # 🔊 VOICE AI
        smart_voice(alert, suggestion)

        alert_data = {
            "alert": alert,
            "suggestion": suggestion,
            "object": obj_name,
            "distance": obj_dist,
            "objects": objects_list[:5],
            "total_objects": total_objects,
            "vehicles": vehicle_count,
            "persons": person_count,
            "direction": direction,
            "crowd": crowd,
            "risk_percent": 100 if closest_distance < 5 else 60 if closest_distance < 10 else 20
        }

        cv2.putText(frame, alert, (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('driver.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/alert')
def alert():
    return jsonify(alert_data)


if __name__ == "__main__":
    app.run(port=5001, debug=True)