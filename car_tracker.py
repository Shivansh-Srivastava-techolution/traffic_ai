import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# Constants
VEHICLE_CLASSES = [1, 2, 3, 5, 6, 7]  # COCO IDs: bicycle, car, motorcycle, bus, train, truck
PIXEL_TO_METER_RATIO = 0.06
REAL_VEHICLE_HEIGHT = 1.5
SPEED_SMOOTHING_FRAMES = 60

# Global storage for tracking
previous_positions = {}
tracking_paths = {}
speed_history = {}

# Load YOLOv8 model
model = YOLO("yolo11x.pt")

def initialize_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, fps, width, height


def draw_box_with_label(frame, box, label, color=(0, 0, 255)):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def calculate_speed(tracker_id, current_pos, bbox_height, fps):
    speed_kmph = 0.0

    if bbox_height > 0:
        pixel_to_meter = REAL_VEHICLE_HEIGHT / bbox_height
    else:
        pixel_to_meter = PIXEL_TO_METER_RATIO  # fallback

    if tracker_id in previous_positions:
        prev_pos = previous_positions[tracker_id]
        pixel_dist = np.linalg.norm(np.array(current_pos) - np.array(prev_pos))
        meters_moved = pixel_dist * pixel_to_meter
        speed_mps = meters_moved * fps
        speed_kmph = speed_mps * 3.6

    previous_positions[tracker_id] = current_pos

    if tracker_id not in speed_history:
        speed_history[tracker_id] = deque(maxlen=SPEED_SMOOTHING_FRAMES)
    speed_history[tracker_id].append(speed_kmph)

    return np.mean(speed_history[tracker_id])


def update_tracking_path(tracker_id, current_pos):
    if tracker_id not in tracking_paths:
        tracking_paths[tracker_id] = []
    tracking_paths[tracker_id].append(current_pos)


def update_heatmap(mask, cx, cy, width, height):
    if 0 <= cy < height and 0 <= cx < width:
        cv2.circle(mask, (cx, cy), radius=8, color=25, thickness=-1)


def process_frame(frame, fps, heatmap_mask, width, height):
    results = model.track(frame, persist=True, conf=0.2, iou=0.5)[0]

    if results.boxes is None:
        return frame

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in VEHICLE_CLASSES:
            continue

        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0]
        tracker_id = int(box.id[0]) if box.id is not None else -1

        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        current_pos = (cx, cy)

        bbox_height = y2 - y1

        smoothed_speed = calculate_speed(tracker_id, current_pos, bbox_height, fps)
        update_tracking_path(tracker_id, current_pos)
        update_heatmap(heatmap_mask, cx, cy, width, height)

        label = f"#{tracker_id} {model.names[cls_id]} {conf:.2f} | {int(smoothed_speed)} km/h"
        draw_box_with_label(frame, (x1, y1, x2, y2), label)

    return frame


def generate_and_display_heatmap(base_frame, heatmap_mask):
    heatmap_color = cv2.applyColorMap(heatmap_mask, cv2.COLORMAP_JET)
    heatmap_blend = cv2.addWeighted(base_frame, 0.6, heatmap_color, 0.4, 0)
    cv2.imwrite("vehicle_path_heatmap.jpg", heatmap_blend)
    cv2.imshow("Heatmap Overlay", heatmap_blend)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(video_path):
    cap, fps, width, height = initialize_video(video_path)
    heatmap_mask = np.zeros((height, width), dtype=np.uint8)
    first_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if first_frame is None:
            first_frame = frame.copy()

        processed_frame = process_frame(frame, fps, heatmap_mask, width, height)
        cv2.imshow("Vehicle Tracker with Smoothed Speed", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if first_frame is not None:
        generate_and_display_heatmap(first_frame, heatmap_mask)


if __name__ == "__main__":
    main("test1.mp4")
