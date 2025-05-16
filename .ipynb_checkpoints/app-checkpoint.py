import sys
sys.modules['torch.classes'] = None  # Prevent Streamlit watcher from introspecting it

import os
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2
import streamlit as st
from collections import deque
import tempfile

from utils.video import initialize_video, draw_box_with_label
from utils.tracking import calculate_speed, update_tracking_path
from utils.model import load_model
from utils.gemini import threaded_gemini_call, get_traffic_condition_text, get_incident_warning_text, set_default_traffic_status
from utils.heatmap import update_heatmap, render_heatmap, make_kernel, SIGMA

os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

# Fonts
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
emoji_font = ImageFont.truetype(font_path, 20)

# Model and globals
model = load_model()
vehicle_frame_map = {}
latest_vehicle_data = deque(maxlen=50)

# Streamlit setup
if "processing_done" not in st.session_state:
    st.session_state.processing_done = False

st.set_page_config(page_title="🚦 Live Traffic Monitor", layout="wide")
st.title("🚗 Live Vehicle Tracking and Heatmap Viewer")

uploaded_file = st.sidebar.file_uploader("Upload a traffic video (.mp4)", type=["mp4"])
start_button = st.sidebar.button("Start Processing")
reset_button = st.sidebar.button("Reset")

st.sidebar.subheader("Settings")
display_update_interval_secs = int(st.sidebar.slider("Display update interval (s)", 1, 10, 3))
speed_offset = int(st.sidebar.number_input("Speed offset (km/h)", value=0))
gemini_model_to_use = st.sidebar.selectbox("Gemini model", ("gemini-2.0-flash", "gemini-2.5-flash-preview-04-17", "gemini-1.5-flash-002"))
average_vehicle_height = int(st.sidebar.number_input("Average vehicle height (m)", value=1))

video_placeholder = st.empty()
heatmap_placeholder = st.empty()
status_text = st.empty()

if reset_button:
    st.session_state.processing_done = False
    st.rerun()

def draw_left_speed_bar(frame, speeds, bar_width=200, title="Top Speeds"):
    h, w = frame.shape[:2]
    bar = np.zeros((h, bar_width, 3), dtype=np.uint8)
    cv2.putText(bar, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    for i, spd in enumerate(sorted(speeds, reverse=True)[:10]):
        y = 60 + i * 30
        cv2.putText(bar, f"{i+1}. {spd} km/h", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    return np.hstack((bar, frame))

def draw_top_bar_with_emoji(frame, incident_text, traffic_text, bar_height, incident_font, traffic_font):
    h, w = bar_height, frame.shape[1]
    bar = np.zeros((h, w, 3), dtype=np.uint8)
    pil = Image.fromarray(bar)
    draw = ImageDraw.Draw(pil)
    txt = incident_text.replace("True", "✔").replace("False", "✕").replace(" | ", "\n")
    y = 5
    for line in txt.split("\n"):
        x0, y0, x1, y1 = draw.textbbox((0,0), line, font=incident_font)
        draw.text((10, y), line, font=incident_font, fill=(255,255,255))
        y += (y1 - y0) + 4
    draw.text((10, y), f"Traffic Conditions: {traffic_text}", font=traffic_font, fill=(255, 255, 0))
    return np.array(pil)

def process_frame(frame, frame_counter, fps, heatmap_accum, gauss_kernel, width, height):
    if frame.shape[:2] != (height, width):
        frame = cv2.resize(frame, (width, height))
    results = model.track(frame, persist=True, conf=0.2, iou=0.5)[0]
    vehicle_data = []
    speeds = []
    for box in results.boxes or []:
        cls_id = int(box.cls[0])
        bbox = box.xyxy[0].tolist()
        tracker_id = int(box.id[0]) if box.id is not None else -1
        if tracker_id not in vehicle_frame_map:
            vehicle_frame_map[tracker_id] = []
        vehicle_frame_map[tracker_id].append((frame_counter, bbox))
        if cls_id not in [1, 2, 3, 5, 6, 7]:
            continue
        x1, y1, x2, y2 = bbox
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        bbox_height = y2 - y1
        speed = calculate_speed(tracker_id, (cx, cy), bbox_height, fps, average_vehicle_height)
        update_tracking_path(tracker_id, (cx, cy))
        update_heatmap(heatmap_accum, cx, cy, 1.0, gauss_kernel)
        label = f"#{tracker_id} {model.names[cls_id]} | {int(speed)} km/h"
        draw_box_with_label(frame, bbox, label)
        vehicle_data.append(label)
        speeds.append(int(speed))
    return frame, vehicle_data, speeds

if start_button and uploaded_file and not st.session_state.processing_done:
    model = load_model()
    st.session_state.processing_done = True

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()

    cap, fps, width, height = initialize_video(tfile.name)
    ret, base_frame = cap.read()
    if not ret:
        st.error("❌ Could not read uploaded video.")
        st.stop()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    set_default_traffic_status()
    heatmap_accum = np.zeros((height, width), dtype=np.float32)
    gauss_kernel = make_kernel(SIGMA) * 20
    frame_counter = 0
    padding_height = 250
    latest_vehicle_data.clear()

    output_path = "temp.webm"
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    LEFT_BAR_W = 200
    out = cv2.VideoWriter(output_path, fourcc, fps, (width + LEFT_BAR_W, height + padding_height))

    if fps == 0:
        fps = 30

    display_update_interval_frames = int(fps * display_update_interval_secs)
    displayed_traffic_text = get_traffic_condition_text()
    displayed_incident_text = get_incident_warning_text()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame, vehicle_data, speeds = process_frame(frame, frame_counter, fps, heatmap_accum, gauss_kernel, width, height)
        latest_vehicle_data.append("; ".join(vehicle_data))
        metadata_text = "\n".join(list(latest_vehicle_data)[-5:])
        if frame_counter % int(fps) == 0:
            threaded_gemini_call(processed_frame.copy(), metadata_text)
        if frame_counter % display_update_interval_frames == 0:
            displayed_traffic_text = get_traffic_condition_text()
            displayed_incident_text = get_incident_warning_text()
        incident_font = ImageFont.truetype(font_path, 28)
        traffic_font = ImageFont.truetype(font_path, 36)
        top_bar = draw_top_bar_with_emoji(processed_frame, displayed_incident_text, displayed_traffic_text, padding_height, incident_font, traffic_font)
        frame_with_bar = np.vstack((top_bar, processed_frame))
        final_frame = draw_left_speed_bar(frame_with_bar, speeds, bar_width=LEFT_BAR_W)
        out.write(final_frame)
        frame_counter += 1

    cap.release()
    out.release()

    st.success("✅ Video processing complete!")

    with open(output_path, "rb") as vid_file:
        vid_byte = vid_file.read()
        st.video(vid_byte)
        st.download_button("⬇️ Download Processed Video", vid_byte, "processed_traffic_video.webm", mime="video/webm")

    heatmap = render_heatmap(heatmap_accum, base_frame=base_frame)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    st.image(heatmap_rgb, channels="RGB", caption="📊 Traffic Density Heatmap", use_column_width=True)
        
    print(vehicle_frame_map)
            
    # === Zoomed video generation section ===
    st.subheader("🔍 Zoom into a specific vehicle")
    tracked_ids = list(vehicle_frame_map.keys())

    if "zoom_ready" not in st.session_state:
        st.session_state.zoom_ready = False
    if "selected_id" not in st.session_state:
        st.session_state.selected_id = None

    if tracked_ids:
        selected_id = st.selectbox("🚗 Select Vehicle ID", tracked_ids)
        if st.button("Prepare Zoom Video"):
            st.session_state.zoom_ready = True
            st.session_state.selected_id = selected_id
    else:
        st.info("ℹ️ No vehicles were detected to zoom into.")

    # === Only generate if zoom_ready ===
    if st.session_state.zoom_ready and st.session_state.selected_id is not None:
        selected_id = st.session_state.selected_id
        st.write(f"Generating zoomed video for vehicle #{selected_id}...")
        cap_zoom = cv2.VideoCapture(tfile.name)
        zoom_output_path = "zoomed_output.webm"
        out_zoom = cv2.VideoWriter(zoom_output_path, fourcc, fps, (400, 400))

        for frame_num, bbox in vehicle_frame_map[selected_id]:
            cap_zoom.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, full_frame = cap_zoom.read()
            if not ret or full_frame is None:
                continue
            x1, y1, x2, y2 = map(int, bbox)
            pad = 20
            x1, y1 = max(x1 - pad, 0), max(y1 - pad, 0)
            x2, y2 = min(x2 + pad, full_frame.shape[1]), min(y2 + pad, full_frame.shape[0])
            zoom_crop = full_frame[y1:y2, x1:x2]
            zoom_resized = cv2.resize(zoom_crop, (400, 400))
            out_zoom.write(zoom_resized)

        cap_zoom.release()
        out_zoom.release()

        st.success("🎥 Zoomed video generated!")

        with open(zoom_output_path, "rb") as zoom_vid:
            zoom_bytes = zoom_vid.read()
            st.video(zoom_bytes)
            st.download_button("⬇️ Download Zoomed Video", zoom_bytes, f"zoomed_vehicle_{selected_id}.webm", mime="video/webm", key="zoomed_download")
