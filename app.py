
import streamlit as st
import os, tempfile, zipfile, subprocess, json
from pathlib import Path

st.set_page_config(page_title="Namaste Phone Editor", page_icon="🎬", layout="centered")
st.title("🎬 Namaste Phone Editor")
st.write("Audio + manhwa panels upload karo. App narration ko scenes me divide karke panels ka rough-cut video banayega.")

api_key = st.text_input("OpenAI API Key", type="password")
audio = st.file_uploader("1️⃣ English narration audio", type=["mp3","wav","m4a","aac","mp4"])
panels = st.file_uploader("2️⃣ Manhwa panels (multiple images)", type=["jpg","jpeg","png","webp"], accept_multiple_files=True)
seconds = st.slider("Panel duration (seconds)", 2, 10, 5)

if st.button("🚀 Video Banao", use_container_width=True):
    if not audio or not panels:
        st.error("Audio aur panels dono upload karo.")
        st.stop()
    if not api_key:
        st.error("OpenAI API key daalo.")
        st.stop()

    st.info("Phone par upload hone ke baad processing server par hogi. Long videos me time lag sakta hai.")

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        work = Path(tempfile.mkdtemp())
        audio_path = work / audio.name
        audio_path.write_bytes(audio.getvalue())

        panel_dir = work / "panels"
        panel_dir.mkdir()
        for i, p in enumerate(panels):
            ext = Path(p.name).suffix.lower() or ".jpg"
            (panel_dir / f"{i:04d}{ext}").write_bytes(p.getvalue())

        st.write("🎙️ Audio transcript ho raha hai...")
        with open(audio_path, "rb") as f:
            tr = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json"
            )

        segments = getattr(tr, "segments", None) or []
        if not segments:
            st.error("Transcript segments nahi mile.")
            st.stop()

        # Simple deterministic panel sequence. This is intentionally lightweight for phone hosting.
        image_files = sorted([p for p in panel_dir.iterdir() if p.is_file()])
        duration = float(segments[-1].end) if hasattr(segments[-1], "end") else len(segments) * seconds

        # Create concat file and normalized images. Requires ffmpeg on the hosting server.
        norm = work / "norm"
        norm.mkdir()
        valid = []
        for i, p in enumerate(image_files):
            out = norm / f"panel_{i:04d}.jpg"
            subprocess.run([
                "ffmpeg","-y","-i",str(p),"-vf",
                "scale=1920:1080:force_original_aspect_ratio=decrease,"
                "pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
                "-q:v","2",str(out)
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            valid.append(out)

        # Repeat panels across the whole audio.
        concat = work / "concat.txt"
        with open(concat, "w", encoding="utf-8") as f:
            t = 0.0
            i = 0
            while t < duration:
                p = valid[i % len(valid)]
                f.write(f"file '{p.as_posix()}'\n")
                f.write(f"duration {seconds}\n")
                t += seconds
                i += 1
            f.write(f"file '{valid[-1].as_posix()}'\n")

        out = work / "namaste_edit.mp4"
        subprocess.run([
            "ffmpeg","-y","-f","concat","-safe","0","-i",str(concat),
            "-i",str(audio_path),"-t",str(duration),
            "-c:v","libx264","-preset","veryfast","-pix_fmt","yuv420p",
            "-c:a","aac","-b:a","192k","-shortest",str(out)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        st.success("✅ Video ready!")
        st.video(str(out))
        st.download_button("⬇️ Download MP4", out.read_bytes(), "namaste_edit.mp4", "video/mp4")
    except Exception as e:
        st.error("Error: " + str(e))
        st.caption("Phone hosting ke liye server par FFmpeg available hona zaroori hai.")
