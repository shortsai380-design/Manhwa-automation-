import streamlit as st
import tempfile
import subprocess
import base64
import json
import re
from pathlib import Path
from openai import OpenAI


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Manhwa AI Automation",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Manhwa AI Automation")

st.write(
    "English narration + manhwa panels upload karo. "
    "AI panels ko analyze karke narration ke according "
    "automatic rough-cut video banayega."
)


# =========================================================
# INPUTS
# =========================================================

api_key = st.text_input(
    "OpenAI API Key",
    type="password"
)

audio = st.file_uploader(
    "🎙️ English narration",
    type=["mp3", "wav", "m4a", "aac", "mp4"]
)

panels = st.file_uploader(
    "🖼️ Manhwa panels",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)


# =========================================================
# SETTINGS
# =========================================================

vision_model = st.selectbox(
    "🧠 Panel AI model",
    [
        "gpt-5.6-luna",
        "gpt-5.6-terra",
        "gpt-5.6-sol"
    ],
    index=0
)

st.caption(
    "Luna = lower-cost option. "
    "Sol = stronger reasoning but potentially higher cost."
)


# =========================================================
# HELPER: IMAGE -> DATA URL
# =========================================================

def image_to_data_url(path):

    suffix = path.suffix.lower()

    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp"
    }.get(suffix, "image/jpeg")

    data = base64.b64encode(
        path.read_bytes()
    ).decode("utf-8")

    return f"data:{mime};base64,{data}"


# =========================================================
# HELPER: EXTRACT JSON
# =========================================================

def extract_json(text):

    text = text.strip()

    # Remove markdown code fences
    text = re.sub(
        r"```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```\s*",
        "",
        text
    )

    # Try direct JSON
    try:
        return json.loads(text)
    except:
        pass

    # Try finding JSON object
    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL
    )

    if match:

        try:
            return json.loads(
                match.group(0)
            )
        except:
            pass

    return None


# =========================================================
# AI: ANALYZE PANELS
# =========================================================

def analyze_panel_batch(client, panel_paths, model):

    content = [
        {
            "type": "input_text",
            "text": """
You are analyzing manga/manhwa panels for an automated
video editor.

For EVERY image, identify:

1. panel_id
2. characters visible
3. important actions
4. emotions/reactions
5. location/background
6. important objects
7. what is happening in the scene
8. a short visual description

IMPORTANT:

- Do NOT invent events that are not visible.
- Focus only on what can actually be seen.
- Keep descriptions concise.
- Preserve the image order.
"""
        }
    ]

    for path in panel_paths:

        data_url = image_to_data_url(path)

        content.append({
            "type": "input_text",
            "text": f"IMAGE FILE: {path.name}"
        })

        content.append({
            "type": "input_image",
            "image_url": data_url
        })

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": content
            }
        ]
    )

    return response.output_text


# =========================================================
# AI: MATCH PANELS TO NARRATION
# =========================================================

def match_panels_to_narration(
    client,
    model,
    narration_segments,
    panel_descriptions
):

    narration_text = []

    for i, seg in enumerate(narration_segments):

        narration_text.append(
            f"""
SEGMENT {i + 1}
START: {seg['start']:.3f}
END: {seg['end']:.3f}
DURATION: {seg['duration']:.3f}
TEXT: {seg['text']}
"""
        )

    prompt = f"""
You are an expert manga/manhwa video editor.

We have:

A) narration segments with exact timestamps

B) AI descriptions of manhwa panels.

Your job is to decide which panel(s) should appear
during each narration segment.

IMPORTANT RULES:

1. Use the visual content of the panels.
2. Match panels semantically to what the narrator is saying.
3. Do NOT simply assign panels in numerical order.
4. A panel can stay on screen longer than 5 seconds.
5. A panel can stay on screen less than 5 seconds.
6. If one narration segment is best represented by one panel,
   use one panel for the whole segment.
7. If two or more panels are useful for one narration segment,
   divide the segment duration between them.
8. Do not use a panel for an event that it clearly does not show.
9. Prefer the most visually relevant panel.
10. Avoid unnecessary rapid cuts.
11. The final timeline must
