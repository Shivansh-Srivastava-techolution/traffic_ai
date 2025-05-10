# utils/gemini.py
import cv2
import threading
import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
import numpy as np
from PIL import ImageFont, ImageDraw, Image

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

_traffic_condition_text = "Analyzing traffic..."
_incident_warning_text = "Monitoring for incidents..."

def draw_with_pil(frame, incident_text, traffic_text):
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Use a font that supports emojis
    font = ImageFont.truetype(font_path, 20)

    draw.text((10, 5), f"Incidents: {incident_text}", font=font, fill=(255, 0, 0))
    draw.text((10, 35), f"Traffic: {traffic_text}", font=font, fill=(255, 255, 0))

    return np.array(img_pil)

def get_traffic_condition_text():
    return _traffic_condition_text

def get_incident_warning_text():
    return _incident_warning_text

def set_default_traffic_status():
    global _traffic_condition_text, _incident_warning_text
    _traffic_condition_text = "Analyzing traffic..."
    _incident_warning_text = "Monitoring for incidents..."
    
def convert_booleans_to_emojis(text: str) -> str:
    # Lowercase everything for uniformity
    text = text.lower()
    # Replace true/false with emojis
    text = text.replace("true", "✅").replace("false", "❌")
    return text


def threaded_gemini_call(frame, metadata_text):
    def run():
        global _traffic_condition_text, _incident_warning_text
        try:
            success, encoded_image = cv2.imencode('.jpg', frame)
            if not success:
                return
            image_bytes = encoded_image.tobytes()

            def traffic_task():
                global _traffic_condition_text
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash-preview-04-17",
                        contents=[
                            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                            f"""Traffic image and info: {metadata_text}. 
                            Classify the traffic condition: Light, Moderate or Heavy. 
                            Just return the class"""
                        ]
                    )
                    _traffic_condition_text = response.text.strip()
                except Exception as e:
                    print(f"Traffic Error: {str(e)}")

            def incident_task():
                global _incident_warning_text
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash-preview-04-17",
                        contents=[
                            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                            f"""
                                You are an expert traffic incident detection AI. Analyze the following traffic frame and metadata to detect incidents based on predefined categories.

                                Input Metadata:
                                {metadata_text}

                                Categories to classify:

                                1. Traffic Jam – At least 6 vehicles appear stopped.
                                2. Slow Traffic – At least 10 vehicles are moving very slowly (under 10 km/h).
                                3. Vehicle Crash – A crash or collision is visible. Mention the vehicle type involved (bike, car, truck, bus).
                                4. Emergency Vehicle – An ambulance or firetruck is visible in the frame.
                                5. No Incident – No visible crashes, jams, slowdowns, or emergency vehicles.

                                Rules:

                                - If `"vehicle_crash"` is true, then `"slow_traffic"` or `"traffic_jam"` should also likely be true.
                                - If `"no_incident"` is true, all others must be false.
                                - Be conservative: only return `true` if confident in visual evidence.

                                **Return a in this format exaclty:
                                    
                                Traffic Jam: true/false | Slow Traffic: true/false | Vehicle Crash (type of vehicle): true/false | Emergency Vehicle: true/false | No Incident: true/false
                                
                                Example output
                                Traffic Jam: false | Slow Traffic: true | Vehicle Crash (car): true | Emergency Vehicle: false | No Incident: false
                                
                                Now analyze the current frame and metadata, and return only the text as specified above
                            """
                        ]
                    )
                    _incident_warning_text = response.text

                except Exception as e:
                    print(f"Incident Error: {str(e)}")

            threading.Thread(target=traffic_task, daemon=True).start()
            threading.Thread(target=incident_task, daemon=True).start()
        except Exception as e:
            print(f"Encoding Error: {str(e)}")

    threading.Thread(target=run, daemon=True).start()
