import streamlit as st
from octoai.client import Client
from octoai.errors import OctoAIClientError, OctoAIServerError
from base64 import b64encode, b64decode
from PIL import Image, ExifTags
from io import BytesIO
import requests
import os
import time
import random
import threading
import queue

CLIP_ENDPOINT_URL = "https://octoshop-clip-4jkxk521l3v1.octoai.run"
FACESWAP_ENDPOINT_URL = "https://octoshop-faceswap-4jkxk521l3v1.octoai.run"

NEGATIVE_SD_PROMPT = "((nsfw)), hands, watermark, blurry photo, distortion, low-res, bad quality, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"

caption_list = [
    "Most Likely to Become President",
    "Future CEO",
    "Most Likely to Travel the World",
    "Most Likely to Write a Bestseller",
    "Future Philanthropist",
    "Most Likely to Discover a Cure",
    "Most Likely to Change the World",
    "Future Tech Titan",
    "Most likely to discover a new planet",
    "Most likely to start a successful tech startup",
    "Future Nobel Prize-winning physicist",
    "Most likely to become a famous chef",
    "Future world-renowned architect",
    "Future leader in environmental activism",
    "Most likely to revolutionize the healthcare industry",
    "Most likely to become a bestselling author",
    "Future CEO of a groundbreaking biotech company",
    "Future leader in space exploration",
    "Future pioneer in artificial intelligence research",
    "Most likely to make a significant scientific discovery",
    "Future leader in renewable energy innovation",
    "Most likely to win a Nobel Prize in Medicine",
    "Future world-renowned neuroscientist",
    "Most likely to start a successful e-commerce empire",
    "Future Nobel Prize-winning economist",
    "Most likely to become a famous singer",
    "Future leader in international diplomacy",
    "Future CEO of a successful hospitality chain",
    "Most likely to discover a cure for a major disease",
    "Future Pulitzer Prize-winning poet",
    "Most likely to become a famous dancer",
    "Future leader in artificial intelligence ethics",
    "Most likely to win an Academy Award for Best Director",
    "Future groundbreaking geneticist",
    "Most likely to start their own successful podcast",
    "Future Grammy Award-winning producer",
    "Most likely to become a famous playwright",
    "Future CEO of a pioneering robotics company",
    "Most likely to win a Nobel Prize in Chemistry",
    "Future leader in sustainable agriculture",
    "Most likely to become a famous actor",
    "Future Pulitzer Prize-winning photographer",
    "Most likely to start a successful nonprofit organization",
    "Future Nobel Prize-winning psychologist",
    "Most likely to win an Olympic gold medal in skiing",
    "Future CEO of a major telecommunications company",
    "Most likely to become a famous voice actor",
    "Future leader in global human rights advocacy",
    "Most likely to win a Pulitzer Prize for poetry",
    "Most likely to start their own successful beauty brand",
    "Future Nobel Prize-winning biologist",
    "Most likely to become a famous magician",
    "Future CEO of a groundbreaking AI startup",
    "Most likely to win an Academy Award for Best Original Screenplay",
    "Future leader in clean energy technology",
    "Most likely to become a famous impressionist",
    "Most likely to start a successful vineyard",
    "Future Nobel Prize-winning astronomer",
    "Future CEO of a major pharmaceutical company",
    "Future leader in wildlife conservation",
    "Most likely to win a Pulitzer Prize for fiction",
    "Future CEO of a groundbreaking space tourism company",
    "Future leader in sustainable urban planning",
    "Future Nobel Prize-winning economist",
    "Future leader in renewable energy policy",
    "Future CEO of a pioneering biotechnology firm"
]

# SDXL futures
img_q = queue.Queue()

# Streamlit page title
st.set_page_config(layout="wide", page_title="Yearbook photo with OctoAI")

st.write("## Yearbook photo with OctoAI 📷")
st.write("\n\n")
st.write("This is a demo preview, and is not meant for redistribution or production use")

# Sidebar for users to input their OctoAI key
octoai_api_key = st.sidebar.text_input('OctoAI API Token', type='password')
os.environ["OCTOAI_API_TOKEN"] = octoai_api_key

# OctoAI client
oai_client = Client(octoai_api_key)

def rotate_image(image):
    try:
        # Rotate based on Exif Data
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        exif = image._getexif()
        if exif[orientation] == 3:
            image=image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image=image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image=image.rotate(90, expand=True)
        return image
    except:
        return image

def rescale_image(image):
    w, h = image.size

    if w == h:
        return image.resize((1024, 1024))
    else:
        if w > h:
            new_height = h
            new_width = int(h * 1152 / 896 )
        else:
            new_width = w
            new_height = int(w * 1152 / 896)

        left = (w - new_width)/2
        top = (h - new_height)/2
        right = (w + new_width)/2
        bottom = (h + new_height)/2
        image = image.crop((left, top, right, bottom))

        if w > h:
            return image.resize((1152, 896))
        else:
            return image.resize((896, 1152))

def read_image(image):
    buffer = BytesIO()
    image.save(buffer, format="png")
    im_base64 = b64encode(buffer.getvalue()).decode("utf-8")
    return im_base64

def query_clip_interrogator(image_str):
    clip_request = {
        "mode": "fast",
        "image": image_str,
    }
    # Send to CLIP endpoint
    reply = requests.post(
        "{}/predict".format(CLIP_ENDPOINT_URL),
        headers={"Content-Type": "application/json"},
        json=clip_request
    )
    labels = reply.json()["completion"]["labels"]
    short_label = labels.split(',')[0:1]
    return ",".join(short_label)

def query_llm(prompt):
    completion = oai_client.chat.completions.create(
        messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant. Keep your responses short and limited to or two one sentence."
        },
        {
            "role": "user",
            "content": prompt
        }
        ],
        model="meta-llama-3-8b-instruct",
        max_tokens=128,
        temperature=0.1
    )
    generated_text = completion.choices[0].message.content
    return generated_text

def query_sdxl(payload):
    oai_client = Client(octoai_api_key)
    future = oai_client.infer_async(
        endpoint_url="https://image.octoai.run/generate/controlnet-sdxl",
        inputs=payload
    )
    while not oai_client.is_future_ready(future):
        time.sleep(0.05)
    result = oai_client.get_future_result(future)
    image_str = result["images"][0]["image_b64"]
    return image_str

def query_faceswap(src_image, dst_image):
    reply = requests.post(
        "{}/predict".format(FACESWAP_ENDPOINT_URL),
        headers={"Content-type": "application/json"},
        json={
            "src_image": src_image,
            "dst_image": dst_image
        }
    )
    return reply.json()["completion"]["image"]

def octoshop(image, labels):
    # Wrap all of this in a try block
    try:
        # Prepare LLAMA request to perform translation
        caption = random.choice(caption_list)
        llm_prompt = '''
        Human: provide detailed, one-sentence image description of a yearbook photo of a student dressed in 90's clothing. The caption of the yearbook photo is: "{}", so add props in the description that reinforce that caption. Identify the subject details based on the following context while ignoring clothing from that context.
        Context: "{}".
        AI: '''.format(caption, labels)
        print("Prompt: {}".format(llm_prompt))
        transformed_labels = query_llm(llm_prompt)
        print("Transcription by the LLM: {}".format(transformed_labels))

        # Prepare SDXL input
        prompt = "(90s yearbook portrait), {}".format(transformed_labels)
        # Enhance the prompt for the chosen style
        width, height = image.size
        payload = {
            "prompt": prompt,
            "negative_prompt": NEGATIVE_SD_PROMPT,
            "width": width,
            "height": height,
            "style_preset": "base",
            "num_images": 1,
            "steps": 25,
            "cfg_scale": 7.5,
            "use_refiner": True,
            "high_noise_frac": 0.8,
            "sampler": "DPM_PLUS_PLUS_2M_KARRAS",
            "controlnet_preprocess": True,
            "controlnet_image": read_image(image),
            "controlnet": "depth_sdxl",
            "controlnet_conditioning_scale": 0.75,
        }
        gen_image = query_sdxl(payload)
        print("SDXL generation done!")

        gen_image = query_faceswap(read_image(image), gen_image)
        print("Faceswap done!")

        output_image = Image.open(BytesIO(b64decode(gen_image)))
        img_q.put((output_image, caption, transformed_labels))

    except OctoAIClientError as e:
        st.write("Oops something went wrong (client error)!")

    except OctoAIServerError as e:
        st.write("Oops something went wrong (server error)")

    except Exception as e:
        st.write("Oops something went wrong (unexpected error)!")

my_upload = st.file_uploader("Take a snap or upload a photo", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    # Pre-process the image
    input_img = Image.open(my_upload)
    input_img = rotate_image(input_img)
    image = rescale_image(input_img)
    # Send CLIP request
    labels = query_clip_interrogator(read_image(image))
    print("The CLIP labels are: {}".format(labels))

    for i in range(0, 4):
        t = threading.Thread(
            target=octoshop,
            args=(
                image,
                labels
            )
        )
        st.runtime.scriptrunner.add_script_run_ctx(t)
        t.start()
    img_q.join()

    col1, col2, col3, col4 = st.columns(4)
    columns = [col1, col2, col3, col4]

    photo_counter = 0
    while photo_counter < 4:
        image, caption, description = img_q.get()
        columns[photo_counter%4].image(image)
        columns[photo_counter%4].text(caption)
        photo_counter += 1

