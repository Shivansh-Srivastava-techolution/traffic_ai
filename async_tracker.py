import cv2
import numpy as np
import time
import threading
from collections import deque
from ultralytics import YOLO
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Constants
VEHICLE_CLASSES = [1, 2, 3, 5, 6, 7]
PIXEL_TO_METER_RATIO = 0.02
REAL_VEHICLE_HEIGHT = 1.2
SPEED_SMOOTHING_FRAMES = 60

previous_positions, tracking_paths, speed_history = {}, {}, {}
model = YOLO("yolo11s.pt")

# Gemini Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
traffic_condition_text = "Analyzing traffic..."
incident_warning_text = "Monitoring for incidents..."
latest_vehicle_data = []

# Video initialization
def initialize_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, fps, width, height

# Drawing boxes and labels
def draw_box_with_label(frame, box, label, color=(0, 0, 255)):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Speed calculation
def calculate_speed(tracker_id, current_pos, bbox_height, fps):
    pixel_to_meter = REAL_VEHICLE_HEIGHT / bbox_height if bbox_height > 0 else PIXEL_TO_METER_RATIO
    speed_kmph = 0.0
    if tracker_id in previous_positions:
        pixel_dist = np.linalg.norm(np.array(current_pos) - np.array(previous_positions[tracker_id]))
        meters_moved = pixel_dist * pixel_to_meter
        speed_kmph = meters_moved * fps * 3.6
    previous_positions[tracker_id] = current_pos
    speed_history.setdefault(tracker_id, deque(maxlen=SPEED_SMOOTHING_FRAMES)).append(speed_kmph)
    return np.mean(speed_history[tracker_id])

# Update tracking paths
def update_tracking_path(tracker_id, current_pos):
    tracking_paths.setdefault(tracker_id, []).append(current_pos)

# Update heatmap mask
def update_heatmap(mask, cx, cy, width, height):
    if 0 <= cy < height and 0 <= cx < width:
        cv2.circle(mask, (cx, cy), radius=8, color=25, thickness=-1)

# Gemini threaded call

def threaded_gemini_call(frame, metadata_text):
    def run():
        try:
            success, encoded_image = cv2.imencode('.jpg', frame)
            if not success:
                return
            image_bytes = encoded_image.tobytes()

            def traffic_task():
                global traffic_condition_text
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash-preview-04-17",
                        contents=[
                            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                            f"Traffic image and info: {metadata_text}. Classify the traffic condition: Light, Moderate or Heavy. Just return the class"
                        ]
                    )
                    traffic_condition_text = response.text.strip()
                except Exception as e:
                    print(f"Error: {str(e)}")

            def incident_task():
                global incident_warning_text
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash-preview-04-17",
                        contents=[
                            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                            f"Based on the following traffic frame and data, detect any abnormal incidents like crashes, collisions, jams, or erratic vehicle behavior. Metadata: {metadata_text}. Just return a brief warning or say 'No incident' make sure to analyze it properly and return warning accordingly."
                        ]
                    )
                    incident_warning_text = response.text.strip()
                except Exception as e:
                    print(f"Error: {str(e)}")

            threading.Thread(target=traffic_task, daemon=True).start()
            threading.Thread(target=incident_task, daemon=True).start()

        except Exception as e:
            print(e)

    threading.Thread(target=run, daemon=True).start()

# Process each frame
def process_frame(frame, fps, heatmap_mask, width, height):
    results = model.track(frame, persist=True, conf=0.7, iou=0.7)[0]
    vehicle_data = []
    for box in results.boxes or []:
        cls_id, conf, bbox, tracker_id = int(box.cls[0]), float(box.conf[0]), box.xyxy[0], int(box.id[0] or -1)
        if cls_id not in VEHICLE_CLASSES:
            continue
        x1, y1, x2, y2 = bbox
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        bbox_height = y2 - y1
        speed = calculate_speed(tracker_id, (cx, cy), bbox_height, fps)
        update_tracking_path(tracker_id, (cx, cy))
        update_heatmap(heatmap_mask, cx, cy, width, height)
        label = f"#{tracker_id} {model.names[cls_id]} {conf:.2f} | {int(speed)} km/h"
        draw_box_with_label(frame, bbox, label)
        vehicle_data.append(label)
    return frame, vehicle_data

# Main execution
def main(video_path):
    global traffic_condition_text, incident_warning_text, latest_vehicle_data
    cap, fps, width, height = initialize_video(video_path)
    heatmap_mask = np.zeros((height, width), dtype=np.uint8)

    frame_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame, vehicle_data = process_frame(frame, fps, heatmap_mask, width, height)
        latest_vehicle_data.append("; ".join(vehicle_data))
        metadata_text = "\n".join(latest_vehicle_data[-5:])

        if frame_counter % 5 == 0:
            threaded_gemini_call(processed_frame.copy(), metadata_text)

        frame_counter += 1
        cv2.putText(processed_frame, f"Traffic Conditions: {traffic_condition_text}", (10, height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(processed_frame, f"Incident Alert: {incident_warning_text}", (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Vehicle Tracker", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("test2.mp4")
