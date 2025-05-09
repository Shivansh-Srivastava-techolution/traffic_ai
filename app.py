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

from utils.video import initialize_video, draw_box_with_label
from utils.tracking import calculate_speed, update_tracking_path
from utils.model import model
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
    st.experimental_rerun()

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

# Main processing block
if start_button and uploaded_file and not st.session_state.processing_done:
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
    latest_vehicle_data.clear()

    # Output video writer
    output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, vehicle_data = process_frame(frame, fps, heatmap_accum, gauss_kernel, width, height)
        latest_vehicle_data.append("; ".join(vehicle_data))
        metadata_text = "\n".join(list(latest_vehicle_data)[-5:])

        if frame_counter % 4 == 0:
            threaded_gemini_call(processed_frame.copy(), metadata_text)

        frame_counter += 1

        # Draw overlays with shadows
        cv2.putText(processed_frame, f"Incident Alert: {get_incident_warning_text()}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(processed_frame, f"Incident Alert: {get_incident_warning_text()}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(processed_frame, f"Traffic Conditions: {get_traffic_condition_text()}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(processed_frame, f"Traffic Conditions: {get_traffic_condition_text()}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        out.write(processed_frame)

    cap.release()
    out.release()

    # Convert to WebM using ffmpeg
    webm_path = output_path.replace(".mp4", ".webm")
    ffmpeg_command = [
        "ffmpeg", "-i", output_path,
        "-c:v", "libvpx-vp9",
        "-b:v", "1M",
        "-c:a", "libopus",
        webm_path
    ]
    subprocess.run(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Display results
    st.success("✅ Video processing complete!")

    with open(webm_path, "rb") as vid_file:
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
