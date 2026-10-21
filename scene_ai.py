
import os
import base64

import cv2
from dotenv import load_dotenv
from openai import OpenAI

# Load API settings from the .env file
load_dotenv()


def get_client():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing. "
            "Add your API key to the .env file."
        )

    return OpenAI(api_key=api_key)


def analyze_image(image_bgr, prompt):
    """
    Analyze an image using a cloud vision model.

    image_bgr: Image from OpenCV in BGR format.
    prompt: Scene description request or user question.
    """

    # Convert the OpenCV image to JPEG
    success, encoded_image = cv2.imencode(
        ".jpg",
        image_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, 85],
    )

    if not success:
        raise ValueError("Could not encode the image.")

    # Encode the image for the API
    image_base64 = base64.b64encode(
        encoded_image.tobytes()
    ).decode("utf-8")

    client = get_client()

    # You can change this to a vision-capable model
    # available to your API account.
    model_name = os.getenv("OPENAI_MODEL", "gpt-5.5")

    response = client.responses.create(
        model=model_name,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You are Vision AI, a visual assistant. "
                            "Describe only what is supported by the "
                            "image. State uncertainty when needed. "
                            "Do not identify people or infer identities. "
                            "Answer the following request:\n"
                            + prompt
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": (
                            "data:image/jpeg;base64,"
                            + image_base64
                        ),
                        "detail": "high",
                    },
                ],
            }
        ],
    )

    return response.output_text
