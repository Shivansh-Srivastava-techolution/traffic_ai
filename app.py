import streamlit as st
import cv2
import numpy as np
from collections import deque
from PIL import Image
import time
import tempfile

from utils.video import initialize_video, draw_box_with_label
from utils.tracking import calculate_speed, update_tracking_path
from utils.model import model
from utils.gemini import threaded_gemini_call, get_traffic_condition_text, get_incident_warning_text, set_default_traffic_status
from utils.heatmap import update_heatmap, render_heatmap, make_kernel, SIGMA

# UI Setup
st.set_page_config(page_title="🚦 Live Traffic Monitor", layout="wide")
st.title("🚗 Live Vehicle Tracking and Heatmap Viewer")

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Upload a traffic video (.mp4)", type=["mp4"])
start_button = st.sidebar.button("Start Processing")

# Display placeholders
video_placeholder = st.empty()
heatmap_placeholder = st.empty()
status_text = st.empty()

latest_vehicle_data = deque(maxlen=50)

def process_frame(frame, fps, heatmap_accum, gauss_kernel, width, height):
    results = model.track(frame, persist=True, conf=0.2, iou=0.5)[0]
    vehicle_data = []
    for box in results.boxes or []:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        bbox   = box.xyxy[0]
        tracker_id = int(box.id[0]) if box.id is not None else -1

        if cls_id not in [1, 2, 3, 5, 6, 7]:
            continue
        x1, y1, x2, y2 = bbox
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        bbox_height = y2 - y1
        speed = calculate_speed(tracker_id, (cx, cy), bbox_height, fps)
        update_tracking_path(tracker_id, (cx, cy))

        weight = 1.0
        update_heatmap(heatmap_accum, cx, cy, weight, gauss_kernel)
        label = f"#{tracker_id} {model.names[cls_id]} {conf:.2f} | {int(speed)} km/h"
        draw_box_with_label(frame, bbox, label)
        vehicle_data.append(label)
    return frame, vehicle_data

if start_button:
    if uploaded_file is None:
        st.warning("Please upload a video to start.")
        st.stop()

    # Save uploaded file to a temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap, fps, width, height = initialize_video(tfile.name)
    ret, base_frame = cap.read()
    if not ret:
        st.error("❌ Could not read the uploaded video.")
        st.stop()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    set_default_traffic_status()

    heatmap_accum = np.zeros((height, width), dtype=np.float32)
    gauss_kernel = make_kernel(SIGMA) * 20
    frame_counter = 0

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
        cv2.putText(processed_frame, f"Traffic Conditions: {get_traffic_condition_text()}",
                    (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(processed_frame, f"Incident Alert: {get_incident_warning_text()}",
                    (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
        status_text.text(f"✅ Processed frame {frame_counter}")

        time.sleep(1 / fps)

    cap.release()

    # Final heatmap
    heatmap = render_heatmap(heatmap_accum, base_frame=base_frame)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap_placeholder.image(heatmap_rgb, channels="RGB", caption="📊 Traffic Density Heatmap")
