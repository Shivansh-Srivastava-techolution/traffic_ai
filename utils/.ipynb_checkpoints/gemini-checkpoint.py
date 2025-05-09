# utils/gemini.py
import cv2
import threading
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

_traffic_condition_text = "Analyzing traffic..."
_incident_warning_text = "Monitoring for incidents..."

def get_traffic_condition_text():
    return _traffic_condition_text

def get_incident_warning_text():
    return _incident_warning_text

def set_default_traffic_status():
    global _traffic_condition_text, _incident_warning_text
    _traffic_condition_text = "Analyzing traffic..."
    _incident_warning_text = "Monitoring for incidents..."

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
                            f"""Based on the following traffic frame and data, detect any abnormal incidents like crashes, collisions, jams, or erratic vehicle behavior. 
                            Metadata: {metadata_text}. 
                            
                            Make sure not to give false alarms. Give more weightage to "no incident"
                            
                            Classify in five categories:
                            1. Traffic Jam - in case of a jam and only if atleast 6 vehicles have stopped
                            2. Slow traffic - if atleast 10 vehicles are very slow (less than 5-10 kmph)
                            3. Vehicle Crash - if there is an accident or crash of the vehicles, tell which vehicle has crashed(bike/car/truck etc.)
                            4. Emergency Vehicle - ambulance/firetruck
                            5. No Incident
                            
                            Just return the class detected"""
                        ]
                    )
                    _incident_warning_text = response.text.strip()
                except Exception as e:
                    print(f"Incident Error: {str(e)}")

            threading.Thread(target=traffic_task, daemon=True).start()
            threading.Thread(target=incident_task, daemon=True).start()
        except Exception as e:
            print(f"Encoding Error: {str(e)}")

    threading.Thread(target=run, daemon=True).start()
