import cv2
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()

def image_inference_cv2(client: genai.Client, cv2_img: cv2.Mat, prompt: str):
    # Encode the cv2 image as JPEG
    success, encoded_image = cv2.imencode('.jpg', cv2_img)
    if not success:
        raise ValueError("Failed to encode image")

    image_bytes = encoded_image.tobytes()
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-04-17",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            prompt
        ]
    )

    print(response.text)

if __name__ == "__main__":
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    cv2_image = cv2.imread("lanes_detected.jpg")
    prompt = "can you give a bbox of the region containing the road in the whole image"
    
    start = time.perf_counter()
    image_inference_cv2(client, cv2_image, prompt)
    endtime = time.perf_counter()

    print(f"Time taken: {endtime - start}")
