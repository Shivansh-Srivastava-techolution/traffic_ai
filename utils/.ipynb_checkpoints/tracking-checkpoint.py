# utils/tracking.py
import numpy as np
from collections import deque

PIXEL_TO_METER_RATIO = 0.1
REAL_VEHICLE_HEIGHT = 1.4
SPEED_SMOOTHING_FRAMES = 120

previous_positions = {}
tracking_paths = {}
speed_history = {}

def calculate_speed(tracker_id, current_pos, bbox_height, fps):
    pixel_to_meter = 1.4 / bbox_height if bbox_height > 0 else 0.05
    speed_kmph = 0.0
    if tracker_id in previous_positions:
        pixel_dist = np.linalg.norm(np.array(current_pos) - np.array(previous_positions[tracker_id]))
        meters_moved = pixel_dist * pixel_to_meter
        speed_kmph = meters_moved * fps * 3.6

    previous_positions[tracker_id] = current_pos
    speed = float(speed_kmph)  # Ensure it's a native Python float
    speed_history.setdefault(tracker_id, deque(maxlen=120)).append(speed)
    return np.mean(speed_history[tracker_id])


def update_tracking_path(tracker_id, current_pos):
    tracking_paths.setdefault(tracker_id, []).append(current_pos)

# Store mask in float32 for better accumulation
# def update_heatmap(mask, cx, cy, width, height):
#     sigma = 20
#     size = 6 * sigma + 1
#     center = size // 2

#     x = np.linspace(-center, center, size)
#     y = np.linspace(-center, center, size)
#     x_grid, y_grid = np.meshgrid(x, y)
#     kernel = np.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
#     kernel = (kernel / kernel.max() * 255).astype(np.uint8)

#     top = max(0, cy - center)
#     left = max(0, cx - center)
#     bottom = min(height, cy + center + 1)
#     right = min(width, cx + center + 1)

#     k_top = center - (cy - top)
#     k_left = center - (cx - left)
#     k_bottom = k_top + (bottom - top)
#     k_right = k_left + (right - left)

#     roi = mask[top:bottom, left:right].astype(np.uint16)
#     roi += kernel[k_top:k_bottom, k_left:k_right].astype(np.uint16)
#     np.clip(roi, 0, 255, out=roi)
#     mask[top:bottom, left:right] = roi.astype(np.uint8)
