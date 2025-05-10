# utils/model.py
from ultralytics import YOLO

def load_model():
    return YOLO("weights/yolo11s.pt")
