from base64 import b64encode
from functools import cache
from io import BytesIO
from typing import Literal, NotRequired, TypedDict

from dotenv import load_dotenv
from PIL.Image import Image
from promplate import Message
from promplate.llm.openai import ChatGenerate

load_dotenv(override=True)


def pil_to_data_url(image: Image, format: str = "PNG"):
    buffered = BytesIO()

    image.save(buffered, format=format)

    img_byte = buffered.getvalue()

    img_base64 = b64encode(img_byte).decode("utf-8")

    return f"data:image/{format.lower()};base64,{img_base64}"


class TextChunk(TypedDict):
    type: Literal["text"]
    text: str


class ImageURL(TypedDict):
    url: str
    detail: NotRequired[Literal["auto", "low", "high"]]


class ImageChunk(TypedDict):
    type: Literal["image_url"]
    image_url: ImageURL


class ExtendedMessage(Message):
    content: str | list[TextChunk | ImageChunk]  # type: ignore


@cache
def get_llm():
    return ChatGenerate().bind(model="gpt-4o-2024-11-20", temperature=0)


def ask_llm(prompt: str, image: Image, information: str):
    messages: list[ExtendedMessage] = [
        {
            "role": "system",
            "content": "Please find an element in the following image based on the user's prompt. You prefer `icon` over `text` boxes.\n\nAlways respond in the language that the user used!!!!!!!!!",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": pil_to_data_url(image), "detail": "high"},
                },
                {"type": "text", "text": f"Here are some OCR results of the images (not perfect):\n\n{information}"},
                {
                    "type": "text",
                    "text": f"Your task is: <prompt>{prompt}</prompt>\n\nPlease find the element (ID) in the image that matches the prompt.",
                },
            ],
        },
    ]

    yield from get_llm()(messages)  # type: ignore
