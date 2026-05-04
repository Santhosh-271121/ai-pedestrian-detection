from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import pyttsx3
import threading
import time

app = Flask(__name__)

# 🔥 MODEL
model = YOLO('yolov8m.pt')

# CAMERA
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)

# 🎤 VOICE ENGINE
engine = pyttsx3.init()
engine.setProperty('rate', 160)

def speak(text):
    def run():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()

# 🔊 SMART VOICE CONTROL (FIXED)
last_status = ""
last_spoken_time = 0

def smart_voice(status):
    global last_status, last_spoken_time

    now = time.time()

    # speak only if changed OR after 4 sec
    if status != last_status or (now - last_spoken_time > 4):

        if status == "DANGER":
            speak("Warning! Vehicle approaching. Do not cross.")
        elif status == "MODERATE":
            speak("Vehicle detected. Please wait.")
        else:
            speak("Safe to cross.")

        last_status = status
        last_spoken_time = now

# 🌙 NIGHT MODE
def enhance_night(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l,a,b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# 📏 DISTANCE
def estimate_distance(h, H):
    return round((H/(h+1))*2,1)

# DATA
data = {
    "status": "SAFE",
    "message": "SAFE TO CROSS",
    "distance": "--",
    "objects": []
}

prev_min_distance = None

# 🎥 MAIN LOOP
def gen_frames():
    global data, prev_min_distance

    while True:
        success, frame = cap.read()
        if not success:
            break

        H, W = frame.shape[:2]

        # 🌙 NIGHT CHECK
        if frame.mean() < 60:
            frame = enhance_night(frame)

        # 🚗 DETECT VEHICLES ONLY
        results = model(frame, conf=0.4, classes=[2,3,5,7])

        objects = []
        distances = []

        if results[0].boxes is not None:

            for box in results[0].boxes.xyxy:

                x1,y1,x2,y2 = map(int, box)

                height = y2 - y1
                distance = estimate_distance(height, H)

                distances.append(distance)

                objects.append({
                    "name": "vehicle",
                    "distance": distance
                })

                # DRAW BOX
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,f"{distance}m",(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)

        # 📉 CLOSEST VEHICLE
        min_distance = min(distances) if distances else None

        # 🚗 APPROACH DETECTION (IMPORTANT FIX)
        approaching = False
        if prev_min_distance is not None and min_distance is not None:
            if min_distance < prev_min_distance:
                approaching = True

        prev_min_distance = min_distance

        # 🚦 STATUS LOGIC (IMPROVED)
        if min_distance is not None:

            if approaching and min_distance < 6:
                status = "DANGER"
                message = "🚫 DO NOT CROSS"

            elif min_distance < 10:
                status = "MODERATE"
                message = "⚠️ WAIT"

            else:
                status = "SAFE"
                message = "✅ SAFE TO CROSS"

        else:
            status = "SAFE"
            message = "✅ NO VEHICLE - SAFE"

        # 🔊 VOICE (SMART)
        smart_voice(status)

        # UPDATE DATA (FOR HTML)
        data = {
            "status": status,
            "message": message,
            "distance": round(min_distance,1) if min_distance else "--",
            "objects": objects[:6]  # limit for UI
        }

        # DISPLAY ON FRAME
        cv2.putText(frame,message,(20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

        # STREAM
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('crossing.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def send_data():
    return jsonify(data)


if __name__ == "__main__":
    app.run(port=5000, debug=True)