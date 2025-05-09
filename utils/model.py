# utils/model.py
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("weights/yolo11s.pt")
