#GPT_vision.py
from fastapi import FastAPI, HTTPException, File, UploadFile
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

# Configure logging
logging.basicConfig(level=logging.INFO)

# Pydantic model for image URL requests
class ImageRequest(BaseModel):
    url: Optional[str] = None

@app.post("/analyze-image/")
async def analyze_image(
    file: Optional[UploadFile] = File(None)
):
    try:
        # Read the image file and encode it to base64
        contents = await file.read()
        encoded_image = base64.b64encode(contents).decode('utf-8')

        # Try to guess the MIME type from the filename
        mime_type, _ = mimetypes.guess_type(file.filename)
        if not mime_type:
            mime_type = 'application/octet-stream'
        logging.info(f"Using MIME type: {mime_type}")

        # Create a data URL (base64-encoded image with MIME type)
        data_url = f"data:{mime_type};base64,{encoded_image}"

        image_content = {
            "type": "image_url",
            "image_url": {"url": data_url}
        }
        logging.info(f"Converted image file to base64 data URL")
    except Exception as e:
        logging.error(f"Failed to process uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to process uploaded file.")

    # Prepare the messages for OpenAI
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                image_content
            ]
        }
    ]

    try:
        # Send request to OpenAI's ChatCompletion API
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # Ensure this is the correct model name
            messages=messages,
            max_tokens=300
        )
        # Extract the assistant's reply
        assistant_reply = response.choices[0].message.content
        logging.info("Received response from OpenAI.")
        return {"response": assistant_reply}
    except openai.OpenAIError as e:
        logging.error(f"OpenAI API error: {e}")
        raise HTTPException(status_code=500, detail="Error communicating with OpenAI API.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)