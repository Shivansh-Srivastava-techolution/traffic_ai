import sys
sys.modules['torch.classes'] = None  # Prevent Streamlit watcher from introspecting it

import os
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

import streamlit as st
import cv2
import numpy as np
from collections import deque
from PIL import Image
import tempfile
import subprocess

import time

from utils.video import initialize_video, draw_box_with_label
from utils.tracking import calculate_speed, update_tracking_path
from utils.model import load_model

model = load_model()

from utils.gemini import threaded_gemini_call, get_traffic_condition_text, get_incident_warning_text, set_default_traffic_status
from utils.heatmap import update_heatmap, render_heatmap, make_kernel, SIGMA

# Initialize session state
if "processing_done" not in st.session_state:
    st.session_state.processing_done = False

# UI Setup
st.set_page_config(page_title="🚦 Live Traffic Monitor", layout="wide")
st.title("🚗 Live Vehicle Tracking and Heatmap Viewer")

# Sidebar for controls
uploaded_file = st.sidebar.file_uploader("Upload a traffic video (.mp4)", type=["mp4"])
start_button = st.sidebar.button("Start Processing")
reset_button = st.sidebar.button("Reset")

# Display placeholders
video_placeholder = st.empty()
heatmap_placeholder = st.empty()
status_text = st.empty()

latest_vehicle_data = deque(maxlen=50)

# Reset logic
if reset_button:
    st.session_state.processing_done = False
    st.rerun()

def process_frame(frame, fps, heatmap_accum, gauss_kernel, width, height):
    
    if frame.shape[0] != height or frame.shape[1] != width:
        frame = cv2.resize(frame, (width, height))
    
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
        label = f"#{tracker_id} {model.names[cls_id]} | {int(speed)} km/h"
        draw_box_with_label(frame, bbox, label)
        vehicle_data.append(label)
    return frame, vehicle_data

# Main processing block
if start_button and uploaded_file and not st.session_state.processing_done:
    model = load_model()  
    
    st.session_state.processing_done = True  # Prevent re-runs

    # Save uploaded file to a temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()

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
    padding_height = 70
    latest_vehicle_data.clear()

    # Output video writer
    # output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = tempfile.NamedTemporaryFile(suffix=".webm", delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*'VP80')  # VP8 codec for .webm
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height + padding_height))
    
    if fps == 0:
        fps = 30
        
    
    display_update_interval_secs = 3
    display_update_interval_frames = int(fps * display_update_interval_secs)

    displayed_traffic_text = get_traffic_condition_text()
    displayed_incident_text = get_incident_warning_text()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, vehicle_data = process_frame(frame, fps, heatmap_accum, gauss_kernel, width, height)
        latest_vehicle_data.append("; ".join(vehicle_data))
        metadata_text = "\n".join(list(latest_vehicle_data)[-5:])

        if frame_counter % int(fps) == 0:
            threaded_gemini_call(processed_frame.copy(), metadata_text)
            
        # Update displayed text only every 5 seconds
        if frame_counter % display_update_interval_frames == 0:
            displayed_traffic_text = get_traffic_condition_text()
            displayed_incident_text = get_incident_warning_text()

        frame_counter += 1

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2

        # Create black padding (top bar)
        top_bar = np.zeros((padding_height, width, 3), dtype=np.uint8)
        cv2.putText(top_bar, f"Incidents: {displayed_incident_text}",
                    (10, 25), font, scale, (0, 0, 255), thickness)

        cv2.putText(top_bar, f"Traffic Conditions: {displayed_traffic_text}",
                    (10, 55), font, scale, (0, 255, 255), thickness)

        # Combine top bar with the actual frame
        processed_frame = np.vstack((top_bar, processed_frame))

        out.write(processed_frame)

    cap.release()
    out.release()
    
    print("YOLO processing done")
    
    st.success("✅ Video processing complete!")
    
    with open(output_path, "rb") as vid_file:
        video_bytes = vid_file.read()
        st.video(video_bytes)
        st.download_button(
            label="⬇️ Download Processed Video (WebM)",
            data=video_bytes,
            file_name="processed_traffic_video.webm",
            mime="video/webm"
        )


    heatmap = render_heatmap(heatmap_accum, base_frame=base_frame)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap_placeholder.image(heatmap_rgb, channels="RGB", caption="📊 Traffic Density Heatmap")
