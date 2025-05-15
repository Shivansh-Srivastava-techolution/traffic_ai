from utils.video import initialize_video, draw_box_with_label
from utils.tracking import calculate_speed, update_tracking_path
from utils.model import model
from collections import deque
from utils.gemini import threaded_gemini_call, get_traffic_condition_text, get_incident_warning_text
from utils.heatmap import update_heatmap, render_heatmap, make_kernel, SIGMA

import cv2
import numpy as np

latest_vehicle_data = deque(maxlen=50)   

def process_frame(frame, fps, heatmap_accum, gauss_kernel, width, height):
    results = model.track(frame, persist=True, conf=0.2, iou=0.5)[0]
    vehicle_data = []
    for box in results.boxes or []:
        cls_id, conf, bbox, tracker_id = int(box.cls[0]), float(box.conf[0]), box.xyxy[0], int(box.id[0] or -1)
        if cls_id not in [1, 2, 3, 5, 6, 7]:
            continue
        x1, y1, x2, y2 = bbox
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        bbox_height = y2 - y1
        speed = calculate_speed(tracker_id, (cx, cy), bbox_height, fps)
        update_tracking_path(tracker_id, (cx, cy))

        weight = 1.0 / max(bbox_height, 1)   # perspective‑aware weighting (optional)
        update_heatmap(heatmap_accum, cx, cy, weight, gauss_kernel)  
        # update_heatmap(heatmap_mask, cx, cy, width, height)
        label = f"#{tracker_id} {model.names[cls_id]} {conf:.2f} | {int(speed)} km/h"
        draw_box_with_label(frame, bbox, label)
        vehicle_data.append(label)
    return frame, vehicle_data

def main(video_path):
    global latest_vehicle_data
    cap, fps, width, height = initialize_video(video_path)
    frame_counter = 0
    base_frame = cap.read()[1]

    heatmap_accum  = np.zeros((height, width), dtype=np.float32)
    gauss_kernel   = make_kernel(SIGMA)


    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame, vehicle_data = process_frame(frame, fps, heatmap_accum, gauss_kernel, width, height)
        latest_vehicle_data.append("; ".join(vehicle_data))
        metadata_text = "\n".join(list(latest_vehicle_data)[-5:])

        if frame_counter % 3 == 0:
            threaded_gemini_call(processed_frame.copy(), metadata_text)
        frame_counter += 1
        
        # Process the response from Gemini 
        incident_warning = get_incident_warning_text()
        warnings = incident_warning.split('|')
        print("Warnings : ", warnings)
        
        text_to_show = ""
        for warning in warnings:
            text_to_show = text_to_show + warnings + "\n"
        
        
        cv2.putText(processed_frame, f"Traffic Conditions: {get_traffic_condition_text()}", (10, height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(processed_frame, f"Incident Alert: {text_to_show}", (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Vehicle Tracker", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

    overlay = render_heatmap(heatmap_accum, base_frame=base_frame)  # live overlay
    cv2.imshow("Traffic Density", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("videos/test4.mp4")