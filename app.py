import streamlit as st
import tempfile
import subprocess
from pathlib import Path

st.set_page_config(page_title="Manhwa Automation", page_icon="🎬")

st.title("🎬 Manhwa Automation")
st.write("English narration + manhwa panels ko narration timing ke according rough-cut video me convert karo.")

api_key = st.text_input("OpenAI API Key", type="password")

audio = st.file_uploader(
    "🎙️ English narration audio",
    type=["mp3", "wav", "m4a", "aac", "mp4"]
)

panels = st.file_uploader(
    "🖼️ Manhwa panels",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

if st.button("🚀 Video Banao", use_container_width=True):

    if not audio or not panels:
        st.error("Audio aur panels dono upload karo.")
        st.stop()

    if not api_key:
        st.error("OpenAI API key daalo.")
        st.stop()

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        work = Path(tempfile.mkdtemp())

        # Save audio
        audio_path = work / audio.name
        audio_path.write_bytes(audio.getvalue())

        # Save panels
        panel_dir = work / "panels"
        panel_dir.mkdir()

        for i, panel in enumerate(panels):
            ext = Path(panel.name).suffix.lower() or ".jpg"
            path = panel_dir / f"{i:04d}{ext}"
            path.write_bytes(panel.getvalue())

        # -------------------------
        # 1. TRANSCRIBE AUDIO
        # -------------------------

        st.write("🎙️ Audio transcribe ho raha hai...")

        with open(audio_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json"
            )

        segments = getattr(transcript, "segments", None) or []

        if not segments:
            st.error("Audio ke timestamp segments nahi mile.")
            st.stop()

        # -------------------------
        # 2. GET AUDIO TIMELINE
        # -------------------------

        timeline = []

        for segment in segments:
            start = float(segment.start)
            end = float(segment.end)
            text = segment.text.strip()

            if text:
                timeline.append({
                    "start": start,
                    "end": end,
                    "text": text
                })

        if not timeline:
            st.error("Narration timeline create nahi ho payi.")
            st.stop()

        st.success(f"✅ {len(timeline)} narration segments mile.")

        # -------------------------
        # 3. NORMALIZE PANELS
        # -------------------------

        st.write("🖼️ Panels prepare ho rahe hain...")

        image_files = sorted([
            p for p in panel_dir.iterdir()
            if p.is_file()
        ])

        norm = work / "normalized"
        norm.mkdir()

        valid = []

        for i, image in enumerate(image_files):

            output = norm / f"panel_{i:04d}.jpg"

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(image),
                    "-vf",
                    "scale=1920:1080:force_original_aspect_ratio=decrease,"
                    "pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
                    "-q:v",
                    "2",
                    str(output)
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            valid.append(output)

        if not valid:
            st.error("Koi valid panel nahi mila.")
            st.stop()

        # -------------------------
        # 4. MATCH PANELS TO NARRATION
        # -------------------------

        st.write("🧠 Narration ke according panels match ho rahe hain...")

        # Abhi basic sequential matching.
        # IMPORTANT
