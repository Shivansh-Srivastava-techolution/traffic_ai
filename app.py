import sys

sys.modules['torch.classes'] = None  # Prevent Streamlit watcher from introspecting it

import os
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2

os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"
os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

import streamlit as st
import cv2
import numpy as np
from collections import deque
from PIL import Image
import tempfile
import subprocess
import emoji

import time

from utils.video import initialize_video, draw_box_with_label
from utils.tracking import calculate_speed, update_tracking_path
from utils.model import load_model

model = load_model()

from utils.gemini import threaded_gemini_call, get_traffic_condition_text, get_incident_warning_text, \
    set_default_traffic_status
from utils.heatmap import update_heatmap, render_heatmap, make_kernel, SIGMA
font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
emoji_font = ImageFont.truetype(font_path, 20)

def draw_left_speed_bar(frame: np.ndarray, speeds: list[int],
                        bar_width: int = 200,
                        title: str = "Top Speeds") -> np.ndarray:
    """
    Draws a vertical bar of width `bar_width` on the left of `frame`,
    listing the top 10 speeds (descending) from `speeds`.
    """
    h, w = frame.shape[:2]
    bar = np.zeros((h, bar_width, 3), dtype=np.uint8)

    # Title
    cv2.putText(bar, title, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # Sort & draw up to 10
    for i, spd in enumerate(sorted(speeds, reverse=True)[:10]):
        y = 60 + i * 30
        txt = f"{i+1}. {spd} m/h"
        cv2.putText(bar, txt, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Stack it on the left
    return np.hstack((bar, frame))

def draw_top_bar_with_emoji(frame, incident_text, traffic_text,
                            bar_height,
                            incident_font: ImageFont.FreeTypeFont,
                            traffic_font: ImageFont.FreeTypeFont):
    h, w = bar_height, frame.shape[1]
    bar = np.zeros((h, w, 3), dtype=np.uint8)
    pil = Image.fromarray(bar)
    draw = ImageDraw.Draw(pil)

    # replace True/False & split lines
    txt = (incident_text
           .replace("True",  "✔")
           .replace("False", "✕")
           .replace(" | ",  "\n"))

    y = 5
    for line in txt.split("\n"):
        # get the bounding‐box of the rendered text
        x0, y0, x1, y1 = draw.textbbox((0,0), line, font=incident_font)
        line_height = y1 - y0
        draw.text((10, y), line, font=incident_font, fill=(255,255,255))
        y += line_height + 4

    # now the traffic line
    draw.text((10, y),
              f"Traffic Conditions: {traffic_text}",
              font=traffic_font,
              fill=(255, 255, 0))

    return np.array(pil)



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

# Add a different section in the side bar
st.sidebar.subheader("Settings")
display_update_interval_secs = int(st.sidebar.slider("Select how often to update the display (in seconds)", 1, 10, 3))

speed_offset = int(st.sidebar.number_input("Add a defined offset ( kmph ) to the speed that is calculated"))

# Select which model to use 
gemini_model_to_use = st.sidebar.selectbox(
    "Which model would you like to use?",
    ("gemini-2.0-flash", "gemini-2.5-flash-preview-04-17", "gemini-1.5-flash-002"),
)

average_vehicle_height = int(st.sidebar.number_input("Insert the height of an average car ( meters )"))

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
    speeds = []
    
    for box in results.boxes or []:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        bbox = box.xyxy[0]
        tracker_id = int(box.id[0]) if box.id is not None else -1

        if cls_id not in [1, 2, 3, 5, 6, 7]:
            continue
        x1, y1, x2, y2 = bbox
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        bbox_height = y2 - y1
        speed = calculate_speed(tracker_id, (cx, cy), bbox_height, fps, average_vehicle_height)
        update_tracking_path(tracker_id, (cx, cy))

        weight = 1.0
        update_heatmap(heatmap_accum, cx, cy, weight, gauss_kernel)
        label = f"#{tracker_id} {model.names[cls_id]} | {int(speed)} m/h"
        
        ########### DON'T TRACK STATIONARY CARS #########
        # if speed > 10:
        draw_box_with_label(frame, bbox, label)
            
        vehicle_data.append(label)
        speeds.append(int(speed))
        
    return frame, vehicle_data, speeds


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
    padding_height = 250
    latest_vehicle_data.clear()

    # Output video writer
    # output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # output_path = tempfile.NamedTemporaryFile(suffix=".webm", delete=False).name
    output_path = "temp.webm"
    fourcc = cv2.VideoWriter_fourcc(*'VP80')  # VP8 codec for .webm
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height + padding_height))
    LEFT_BAR_W = 200
    out = cv2.VideoWriter(output_path, fourcc, fps, (width + LEFT_BAR_W, height + padding_height))

    if fps == 0:
        fps = 30

    display_update_interval_frames = int(fps * display_update_interval_secs)

    displayed_traffic_text = get_traffic_condition_text()
    displayed_incident_text = get_incident_warning_text()
    
    LEFT_BAR_W = 200
    

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, vehicle_data, speeds = process_frame(frame, fps, heatmap_accum, gauss_kernel, width, height)
        latest_vehicle_data.append("; ".join(vehicle_data))
        metadata_text = "\n".join(list(latest_vehicle_data)[-5:])

        if frame_counter % int(fps) == 0:
            threaded_gemini_call(processed_frame.copy(), metadata_text)

        # Update displayed text only every 5 seconds
        if frame_counter % display_update_interval_frames == 0:
            displayed_traffic_text = get_traffic_condition_text()
            displayed_incident_text = get_incident_warning_text()

        # Process the Incident text
        # Current format :  Traffic Jam: False | Slow Traffic: True | Vehicle Crash : True | Emergency Vehicle: False | No Incident: False
        # Expected Format : Traffic Jam : ✔️ \n Slow Traffic : ✔️ \n Vehicle Crash : ✔️ \n Emergency Vehicle : ✔️ \n No Incident : ✔️
        print("Display Text")
        print(displayed_incident_text)
        
#         check_emoji = str(emoji.emojize(':check_mark:'))
#         cross_emoi = str(emoji.emojize(':cross_mark:'))
        
#         if "✔️" in displayed_incident_text:
#             displayed_incident_text.replace("✔️", check_emoji)
            
#         if "❌" in displayed_incident_text:
#             displayed_incident_text.replace("❌", cross_emoi)
        
        incident_font = ImageFont.truetype(font_path, 28)   # white text at 28 px
        traffic_font = ImageFont.truetype(font_path, 36)   # “Traffic Conditions” at 36 px

                   
        top_bar = draw_top_bar_with_emoji(processed_frame,
                                          displayed_incident_text,
                                          displayed_traffic_text,
                                          bar_height=250,
                                          incident_font=incident_font,
                                          traffic_font=traffic_font)

        # frame_counter += 1

        # font = cv2.FONT_HERSHEY_SIMPLEX
        # scale = 0.6
        # thickness = 2

        # Create black padding (top bar)
#         top_bar_traffic = np.zeros((padding_height, width, 3), dtype=np.uint8)
#         cv2.putText(top_bar_traffic, f"Traffic Conditions: {displayed_traffic_text}",
#                     (10, 55), font, scale, (0, 255, 255), thickness)

#         # Combine top bar with the actual frame
#         top_bar = np.vstack((top_bar_traffic, top_bar_incident))
    
        # Stack top bar
        frame_with_bar = np.vstack((top_bar, processed_frame))
        
        # now add left-hand speeds bar
        final_frame = draw_left_speed_bar(frame_with_bar, speeds, bar_width=LEFT_BAR_W)
        out.write(final_frame)
        
        
        


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
