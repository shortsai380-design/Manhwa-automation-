import streamlit as st
import tempfile
import subprocess
import base64
import json
import math
import mimetypes
import re
from pathlib import Path

from openai import OpenAI
import imageio_ffmpeg


# ============================================================
# MANHWA AUTOMATION - PHONE FRIENDLY STREAMLIT APP
# Pipeline:
# Audio -> transcription timestamps -> panel AI descriptions
# -> narration/panel semantic matching -> dynamic timeline
# -> FFmpeg rough-cut MP4 with original narration audio
# ============================================================

st.set_page_config(
    page_title="Manhwa AI Automation",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 Manhwa AI Automation")
st.caption(
    "Upload English narration + ordered manhwa panels. "
    "The app transcribes the narration, analyzes panels, matches panels to meaning, "
    "assigns dynamic durations, and creates a rough-cut MP4."
)

# -----------------------------
# Helpers
# -----------------------------

def get_ffmpeg():
    return imageio_ffmpeg.get_ffmpeg_exe()


def run_cmd(cmd):
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-5000:])
    return result.stdout


def get_audio_duration(path):
    ffmpeg = get_ffmpeg()
    # ffmpeg can report duration to stderr; using ffprobe is not guaranteed.
    cmd = [
        ffmpeg,
        "-i", str(path),
        "-f", "null",
        "-"
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    text = result.stderr
    import re
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if not m:
        return None
    h, mi, sec = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return h * 3600 + mi * 60 + sec


def image_to_data_url(path):
    suffix = Path(path).suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix, "image/jpeg")

    data = Path(path).read_bytes()
    return f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}"


def clean_json(text):
    """Extract JSON even if the model wrapped it in markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start:end + 1]
    return text


def transcribe_audio(client, audio_path):
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=(Path(audio_path).name, f, "application/octet-stream"),
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    raw = result.model_dump() if hasattr(result, "model_dump") else result
    segments = []

    for i, seg in enumerate(raw.get("segments", []) or []):
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start = float(seg.get("start", 0))
        end = float(seg.get("end", start))
        if end <= start:
            end = start + 0.5
        segments.append({
            "id": i + 1,
            "start": start,
            "end": end,
            "duration": end - start,
            "text": text,
        })

    # Fallback if the API returned only full text.
    if not segments:
        full_text = str(raw.get("text", "")).strip()
        duration = get_audio_duration(audio_path) or 1.0
        if full_text:
            segments = [{
                "id": 1,
                "start": 0.0,
                "end": duration,
                "duration": duration,
                "text": full_text,
            }]

    return segments


def analyze_panels(client, model, panel_paths, batch_size=6):
    """Analyze panels in small batches so phone/cloud uploads remain practical."""
    all_descriptions = []

    for batch_start in range(0, len(panel_paths), batch_size):
        batch = panel_paths[batch_start:batch_start + batch_size]

        content = [{
            "type": "input_text",
            "text": """
You are analyzing ordered manhwa panels for an automatic video editor.

For EACH image, return:
- panel_id: the supplied panel number
- description: concise but specific visual description
- characters: important visible characters and their actions
- setting: location/environment
- emotion: main emotion
- dialogue_or_text: visible speech/text if readable
- story_keywords: 5-10 useful semantic keywords

Do NOT assume anything not visible.
Do NOT write a story summary.
Return ONLY valid JSON in this exact shape:
{
  "panels": [
    {
      "panel_id": 1,
      "description": "...",
      "characters": ["..."],
      "setting": "...",
      "emotion": "...",
      "dialogue_or_text": "...",
      "story_keywords": ["..."]
    }
  ]
}
"""
        }]

        for idx, path in enumerate(batch, start=batch_start + 1):
            content.append({
                "type": "input_text",
                "text": f"PANEL {idx}"
            })
            content.append({
                "type": "input_image",
                "image_url": image_to_data_url(path),
                "detail": "low",
            })

        response = client.responses.create(
            model=model,
            input=[{"role": "user", "content": content}],
        )

        parsed = json.loads(clean_json(response.output_text))
        for item in parsed.get("panels", []):
            try:
                pid = int(item["panel_id"])
            except Exception:
                continue
            item["panel_id"] = pid
            all_descriptions.append(item)

    # Keep original panel order even if model returned them differently.
    by_id = {x["panel_id"]: x for x in all_descriptions}
    ordered = []
    for i in range(1, len(panel_paths) + 1):
        ordered.append(
            by_id.get(i, {
                "panel_id": i,
                "description": "No AI description available.",
                "characters": [],
                "setting": "",
                "emotion": "",
                "dialogue_or_text": "",
                "story_keywords": [],
            })
        )
    return ordered


def match_panels(client, model, segments, panel_descriptions):
    """Ask the model to map each narration segment to one or more panels."""
    panel_text = []
    for p in panel_descriptions:
        panel_text.append(
            f"""PANEL {p['panel_id']}
Description: {p.get('description', '')}
Characters: {', '.join(p.get('characters', []))}
Setting: {p.get('setting', '')}
Emotion: {p.get('emotion', '')}
Visible text: {p.get('dialogue_or_text', '')}
Keywords: {', '.join(p.get('story_keywords', []))}
"""
        )

    segment_text = []
    for s in segments:
        segment_text.append(
            f"""SEGMENT {s['id']}
START: {s['start']:.3f}
END: {s['end']:.3f}
DURATION: {s['duration']:.3f}
NARRATION: {s['text']}
"""
        )

    prompt = f"""
You are the semantic editor for a manhwa recap video.

We have narration segments with exact timestamps and ordered manhwa panels.

Your job is to choose the panel(s) that visually best represent the meaning
of EACH narration segment.

Rules:
1. Match by meaning and visible content, NOT merely by panel order.
2. A panel can be reused when it is genuinely the best visual.
3. Prefer one panel when one panel clearly represents the whole segment.
4. Use multiple panels only when the narration clearly contains multiple visual beats.
5. Do not invent visual events.
6. Do not force a panel just because it is adjacent in the source order.
7. Timing must remain continuous and must cover the entire narration.
8. For multiple panels, give weights that sum to 1.0.
9. A panel should normally remain visible for at least 1.0 second.
10. Avoid rapid cuts. Use fewer panels if the narration does not need a cut.
11. The final video should feel like a human-edited manhwa recap.

Return ONLY valid JSON:
{{
  "matches": [
    {{
      "segment_id": 1,
      "panels": [
        {{"panel_id": 3, "weight": 1.0}}
      ]
    }}
  ]
}}

NARRATION:
{chr(10).join(segment_text)}

PANEL DESCRIPTIONS:
{chr(10).join(panel_text)}
"""

    response = client.responses.create(
        model=model,
        input=prompt,
    )

    data = json.loads(clean_json(response.output_text))
    return data.get("matches", [])


def normalize_matches(segments, matches, panel_count):
    by_segment = {}
    for m in matches:
        try:
            sid = int(m["segment_id"])
        except Exception:
            continue

        panels = []
        for p in m.get("panels", []):
            try:
                pid = int(p["panel_id"])
                weight = float(p.get("weight", 1.0))
            except Exception:
                continue
            if 1 <= pid <= panel_count and weight > 0:
                panels.append({"panel_id": pid, "weight": weight})

        if panels:
            total = sum(x["weight"] for x in panels)
            for x in panels:
                x["weight"] /= total
            by_segment[sid] = panels

    # Fallback: use panel 1 for any missing segment.
    # This keeps the pipeline runnable rather than crashing.
    final = []
    for s in segments:
        panels = by_segment.get(s["id"])
        if not panels:
            panels = [{"panel_id": 1, "weight": 1.0}]
        final.append({
            "segment_id": s["id"],
            "panels": panels,
        })
    return final


def build_timeline(segments, matches):
    match_by_id = {m["segment_id"]: m["panels"] for m in matches}
    timeline = []

    for seg in segments:
        choices = match_by_id.get(seg["id"], [{"panel_id": 1, "weight": 1.0}])
        duration = max(0.1, seg["duration"])

        # Dynamic timing: segment duration is divided according to AI weights.
        # Very short pieces are merged into the strongest panel.
        if duration < 2.0 or len(choices) == 1:
            choices = [max(choices, key=lambda x: x["weight"])]

        total_weight = sum(x["weight"] for x in choices) or 1.0
        cursor = seg["start"]

        for index, choice in enumerate(choices):
            if index == len(choices) - 1:
                end = seg["end"]
            else:
                piece = duration * (choice["weight"] / total_weight)
                end = min(seg["end"], cursor + piece)

            if end - cursor >= 0.35:
                timeline.append({
                    "panel_id": int(choice["panel_id"]),
                    "start": cursor,
                    "end": end,
                    "duration": end - cursor,
                    "segment_id": seg["id"],
                    "text": seg["text"],
                })
                cursor = end

    return timeline


def make_video(panel_paths, timeline, audio_path, output_path):
    """
    Creates a single FFmpeg slideshow stream from the timeline and muxes
    the original narration audio. Every timeline item has its own duration.
    """
    ffmpeg = get_ffmpeg()

    if not timeline:
        raise RuntimeError("Timeline is empty.")

    # Build a filter_complex with one input per timeline item.
    # This is intentionally simple and robust for Streamlit Cloud.
    inputs = []
    filters = []

    for i, item in enumerate(timeline):
        panel_index = item["panel_id"] - 1
        if panel_index < 0 or panel_index >= len(panel_paths):
            raise RuntimeError(f"Invalid panel id: {item['panel_id']}")

        path = str(panel_paths[panel_index])
        duration = max(0.35, float(item["duration"]))

        inputs.extend(["-loop", "1", "-t", f"{duration:.3f}", "-i", path])

        filters.append(
            f"[{i}:v]"
            f"scale=1920:1080:force_original_aspect_ratio=decrease,"
            f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,"
            f"setsar=1,"
            f"format=yuv420p,"
            f"setpts=PTS-STARTPTS[v{i}]"
        )

    concat_inputs = "".join(f"[v{i}]" for i in range(len(timeline)))
    filters.append(
        f"{concat_inputs}concat=n={len(timeline)}:v=1:a=0[outv]"
    )

    cmd = [ffmpeg, "-y"]
    cmd.extend(inputs)
    cmd.extend(["-i", str(audio_path)])
    cmd.extend([
        "-filter_complex", ";".join(filters),
        "-map", "[outv]",
        "-map", f"{len(timeline)}:a:0",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(output_path),
    ])

    run_cmd(cmd)


def format_time(seconds):
    seconds = max(0, float(seconds))
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:05.2f}"


# -----------------------------
# UI
# -----------------------------

api_key = st.text_input(
    "🔑 OpenAI API Key",
    type="password",
    help="Your key is used only for this Streamlit session.",
)

audio_file = st.file_uploader(
    "🎙️ English narration/audio",
    type=["mp3", "wav", "m4a", "aac", "mp4", "mpeg", "mpga", "ogg", "webm"],
)

panel_files = st.file_uploader(
    "🖼️ Manhwa panels — upload in story order",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

st.divider()

vision_model = st.selectbox(
    "🧠 Panel AI model",
    [
        "gpt-5.6-luna",
        "gpt-5.6-terra",
        "gpt-5.6-sol",
    ],
    index=0,
)

st.caption(
    "Luna = lower-cost option. Terra/Sol can be used when you want stronger reasoning."
)

analyze_button = st.button(
    "🚀 Analyze + Match + Build Rough Cut",
    type="primary",
    use_container_width=True,
)

if analyze_button:
    if not api_key:
        st.error("OpenAI API Key enter karo.")
        st.stop()

    if not audio_file:
        st.error("Narration audio upload karo.")
        st.stop()

    if not panel_files:
        st.error("Kam se kam 1 manhwa panel upload karo.")
        st.stop()

    client = OpenAI(api_key=api_key)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        audio_path = tmp_path / audio_file.name
        audio_path.write_bytes(audio_file.getvalue())

        # Preserve uploader order exactly as received.
        panel_paths = []
        for i, uploaded in enumerate(panel_files, start=1):
            suffix = Path(uploaded.name).suffix.lower() or ".jpg"
            p = tmp_path / f"panel_{i:04d}{suffix}"
            p.write_bytes(uploaded.getvalue())
            panel_paths.append(p)

        try:
            # 1. Transcription
            st.subheader("1️⃣ Transcribing narration")
            with st.spinner("Audio transcribe ho raha hai..."):
                segments = transcribe_audio(client, audio_path)

            if not segments:
                st.error("Audio se narration text nahi mila.")
                st.stop()

            st.success(f"{len(segments)} narration segments detected.")

            with st.expander("📜 Transcript"):
                for s in segments:
                    st.write(
                        f"**{format_time(s['start'])} → {format_time(s['end'])}** "
                        f"{s['text']}"
                    )

            # 2. Panel analysis
            st.subheader("2️⃣ Analyzing panels")
            with st.spinner(f"{len(panel_paths)} panels AI se analyze ho rahe hain..."):
                panel_descriptions = analyze_panels(
                    client,
                    vision_model,
                    panel_paths,
                )

            st.success(f"{len(panel_descriptions)} panels analyzed.")

            with st.expander("🖼️ Panel analysis"):
                for p in panel_descriptions:
                    st.write(
                        f"**Panel {p['panel_id']}** — {p['description']}"
                    )

            # 3. Matching
            st.subheader("3️⃣ Matching narration → panels")
            with st.spinner("Meaning-based panel matching ho rahi hai..."):
                raw_matches = match_panels(
                    client,
                    vision_model,
                    segments,
                    panel_descriptions,
                )

            matches = normalize_matches(
                segments,
                raw_matches,
                len(panel_paths),
            )

            # 4. Timeline
            st.subheader("4️⃣ Building dynamic timeline")
            timeline = build_timeline(segments, matches)

            st.success(
                f"{len(timeline)} visual timeline clips created. "
                "Durations narration timestamps se dynamic hain."
            )

            with st.expander("⏱️ Timeline preview"):
                for item in timeline:
                    st.write(
                        f"**{format_time(item['start'])} → "
                        f"{format_time(item['end'])}** "
                        f"| Panel {item['panel_id']} "
                        f"| {item['duration']:.2f}s "
                        f"| {item['text']}"
                    )

            # 5. FFmpeg
            st.subheader("5️⃣ Rendering rough-cut MP4")
            output_path = tmp_path / "manhwa_rough_cut.mp4"

            progress = st.progress(0)
            with st.spinner("FFmpeg video render kar raha hai..."):
                make_video(
                    panel_paths,
                    timeline,
                    audio_path,
                    output_path,
                )
            progress.progress(100)

            if not output_path.exists() or output_path.stat().st_size == 0:
                raise RuntimeError("MP4 file create nahi hui.")

            st.success("🎉 Rough-cut ready!")

            st.video(str(output_path))

            st.download_button(
                "⬇️ Download rough-cut MP4",
                data=output_path.read_bytes(),
                file_name="manhwa_rough_cut.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

            # Timeline JSON download is useful for later editing.
            timeline_json = json.dumps(
                {
                    "segments": segments,
                    "matches": matches,
                    "timeline": timeline,
                },
                ensure_ascii=False,
                indent=2,
            )

            st.download_button(
                "⬇️ Download timeline JSON",
                data=timeline_json,
                file_name="manhwa_timeline.json",
                mime="application/json",
                use_container_width=True,
            )

        except Exception as e:
            st.error("❌ Processing failed.")
            st.exception(e)
            st.info(
                "Agar error API/model related hai, exact error text screenshot ke "
                "saath bhejo. Code ko blindly change mat karna."
            )
