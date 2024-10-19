#GPT_vision.py
import io
from PIL import Image
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.params import Form
from flask.cli import load_dotenv
from pydantic import BaseModel
from typing import Optional
import openai
import os
import logging
import base64
import mimetypes

# Load environment variables from .env file
load_dotenv()

# Set the OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

app = FastAPI()


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Configure logging
logging.basicConfig(level=logging.INFO)

# Pydantic model for image URL requests
class ImageRequest(BaseModel):
    url: Optional[str] = None

def optimize_image(image_file):
    # Open the image file
    with Image.open(image_file) as img:
        # Resize the image (e.g., to 800x800) while maintaining the aspect ratio
        img.thumbnail((800, 800), Image.LANCZOS)  # Use LANCZOS instead of ANTIALIAS
        
        # Create a BytesIO object to hold the image data
        img_byte_arr = io.BytesIO()
        # Save the image as JPEG with quality 85 (adjust as necessary)
        img.save(img_byte_arr, format='JPEG', quality=85)
        img_byte_arr.seek(0)  # Move to the start of the BytesIO buffer
        
        return img_byte_arr

@app.post("/analyze-image/")
async def analyze_image(
    file: Optional[str] = File(None),
    response_history: Optional[str] = Form(None),
    system_prompt: Optional[str] = Form(None)
):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    try:
        # Getting the base64 string
        base64_image = encode_image(file)

        response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "What is in this image?",
                },
                {
                "type": "image_url",
                "image_url": {
                    "url":  f"data:image/jpeg;base64,{base64_image}"
                },
                },
            ],
            }
        ],
        )
        assistant_reply = response.choices[0]
        logging.info("responses: ",response.choices[0])
    except openai.OpenAIError as e:
        logging.error(f"OpenAI API error: {e}")
        raise HTTPException(status_code=500, detail="Error communicating with OpenAI API.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)