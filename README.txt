# Manhwa AI Automation

Phone-friendly Streamlit app for:
1. English narration transcription with timestamps
2. AI analysis of ordered manhwa panels
3. Meaning-based narration -> panel matching
4. Dynamic panel durations based on narration timestamps
5. FFmpeg rough-cut MP4 with original narration audio

## Streamlit deployment

Files:
- app.py
- requirements.txt

Enter your OpenAI API key in the app.

Upload:
- narration audio
- manhwa panels in story order

Then press:
Analyze + Match + Build Rough Cut

The app creates:
- manhwa_rough_cut.mp4
- manhwa_timeline.json

## Important
OpenAI API usage is billed separately from Streamlit hosting.
