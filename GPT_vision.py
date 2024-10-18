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

# Configure logging
logging.basicConfig(level=logging.INFO)

# Pydantic model for image URL requests
class ImageRequest(BaseModel):
    url: Optional[str] = None

def optimize_image(image_file):
    # Open the image file
    with Image.open(image_file) as img:
        # Resize the image (e.g., to 800x800) while maintaining the aspect ratio
        img.thumbnail((800, 800), Image.ANTIALIAS)
        
        # Create a BytesIO object to hold the image data
        img_byte_arr = io.BytesIO()
        # Save the image as JPEG with quality 85 (adjust as necessary)
        img.save(img_byte_arr, format='JPEG', quality=85)
        img_byte_arr.seek(0)  # Move to the start of the BytesIO buffer
        
        return img_byte_arr

@app.post("/analyze-image/")
async def analyze_image(
    file: Optional[UploadFile] = File(None),
    response_history: Optional[str] = Form(None),
    system_prompt: Optional[str] = Form(None)
):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    try:
       
        # Optimize the image
        img_byte_arr = optimize_image(file.file)
        # Read the image file and encode it to base64
        contents = await img_byte_arr.read()
        encoded_image = base64.b64encode(contents).decode('utf-8')

        # Try to guess the MIME type from the filename
        mime_type, _ = mimetypes.guess_type(file.filename)
        if not mime_type:
            mime_type = 'application/octet-stream'
        logging.info(f"Using MIME type: {mime_type}")

        # Create a data URL (base64-encoded image with MIME type)
        data_url = f"data:{mime_type};base64,{encoded_image}"

        image_content = f"Here is the image: {data_url}"
        logging.info("Converted image file to base64 data URL. "+image_content)

    except Exception as e:
        logging.error(f"Failed to process uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to process uploaded file.")

    # Prepare messages for OpenAI API
    messages = []

    # Include system prompt if provided
    if system_prompt:
        if len(system_prompt) > 100: # For rate limit
            messages.append({"role": "system", "content": "You're a helpful and keen image analyst"})
        else:
            messages.append({"role": "system", "content": system_prompt})

    # Include response history if provided
    if response_history:
        messages.append({"role": "user", "content":f"{response_history} {image_content}"})

    # Include user request about the image
    else: 
        messages.append({
            "role": "user",
            "content": f"What's in this image? {image_content}"
        })

    logging.info("Message structure:", messages)
    try:
        # Send request to OpenAI's ChatCompletion API
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Ensure this is the correct model name
            messages=messages,
            max_tokens=500
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