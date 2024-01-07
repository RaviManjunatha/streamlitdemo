import streamlit as st

from vertexai.preview.generative_models import (
    GenerationConfig,
    GenerativeModel,
    Image,
    Part,
)

API_KEY="AIzaSyA3rV2szITkxaWQX4PL8k4qOq7CHqHl7fA"
palm.configure(api_key=API_KEY)

multimodal_model = GenerativeModel("gemini-pro-vision")
import http.client
import typing
import urllib.request

import IPython.display
from PIL import Image as PIL_Image
from PIL import ImageOps as PIL_ImageOps


def display_images(
    images: typing.Iterable[Image],
    max_width: int = 600,
    max_height: int = 350,
) -> None:
    for image in images:
        pil_image = typing.cast(PIL_Image.Image, image._pil_image)
        if pil_image.mode != "RGB":
            # RGB is supported by all Jupyter environments (e.g. RGBA is not yet)
            pil_image = pil_image.convert("RGB")
        image_width, image_height = pil_image.size
        if max_width < image_width or max_height < image_height:
            # Resize to display a smaller notebook image
            pil_image = PIL_ImageOps.contain(pil_image, (max_width, max_height))
        IPython.display.display(pil_image)


def get_image_bytes_from_url(image_url: str) -> bytes:
    with urllib.request.urlopen(image_url) as response:
        response = typing.cast(http.client.HTTPResponse, response)
        image_bytes = response.read()
    return image_bytes


def load_image_from_url(image_url: str) -> Image:
    image_bytes = get_image_bytes_from_url(image_url)
    return Image.from_bytes(image_bytes)


def display_content_as_image(content: str | Image | Part) -> bool:
    if not isinstance(content, Image):
        return False
    display_images([content])
    return True


def display_content_as_video(content: str | Image | Part) -> bool:
    if not isinstance(content, Part):
        return False
    part = typing.cast(Part, content)
    file_path = part.file_data.file_uri.removeprefix("gs://")
    video_url = f"https://storage.googleapis.com/{file_path}"
    IPython.display.display(IPython.display.Video(video_url, width=600))
    return True


def print_multimodal_prompt(contents: list[str | Image | Part]):
    """
    Given contents that would be sent to Gemini,
    output the full multimodal prompt for ease of readability.
    """
    for content in contents:
        if display_content_as_image(content):
            continue
        if display_content_as_video(content):
            continue
        print(content)

advancedeconomies_url = "https://storage.googleapis.com/gemniloandemos/Advancedeconomiesinflation.JPG"
emergingeconomies_url = "https://storage.googleapis.com/gemniloandemos/EmergingEconomiesinflation.JPG"
advancedeconomies = load_image_from_url(advancedeconomies_url)
emergingeconomies = load_image_from_url(emergingeconomies_url)


prompt = """You are an excellent guide for Portfolio Investing. Above are two charts, Image 1 which shows the Inflationary trend of Advanced Economies and Image 2 which shows the Inflation chart of Emerging Economies.  In both the charts, y-axis represent the Inflation percentage and x-axis represents the month of a year.
From the above charts, Suggest me which country can offer me good returns for my investment

Your answer should be based on the following steps,
Step 1:  For which county in the chart  Extract the minimum and maximum Inflation and list them
Step 2 : From the information extracted above list those countries whose minimum and maximum Inflation are between 4% to 8% and recommend these as the countries to invest your clients.
"""

contents = [
    advancedeconomies,
    emergingeconomies,
    prompt

]

responses = multimodal_model.generate_content(contents, stream=True)

st.header("Vertex AI Gemini API", divider="rainbow")

#print("-------Prompt--------")
#print_multimodal_prompt(contents)

print("\n-------Response--------")
for response in responses:
    st.markdown(response.text, end ="") 
