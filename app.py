import plotly.express as px
import pandas as pd
import os
import gradio as gr
import whisper
import datetime
import re
from groq import Groq

# =========================
# MODELS & CLIENTS
# =========================
model = whisper.load_model("base")

client = Groq(api_key="YOUR_API_KEY")

# =========================
# IN-MEMORY STATE
# =========================
schedule_data = {
    "final_schedule": None,
    "has_edited": False
}

# =========================
# HELPERS
# =========================
def extract_dates(text):
    pattern = r'(\d{1,2}:\d{2} [ap]m)\s*-\s*(\d{1,2}:\d{2} [ap]m)'
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None, None

    today = datetime.date.today()
    start = datetime.datetime.combine(
        today, datetime.datetime.strptime(match.group(1), "%I:%M %p").time()
    )
    end = datetime.datetime.combine(
        today, datetime.datetime.strptime(match.group(2), "%I:%M %p").time()
    )
    return start, end


def parse_schedule_to_df(schedule_text):
    pattern = r'(\d{1,2}:\d{2} [ap]m)\s*-\s*(\d{1,2}:\d{2} [ap]m):\s*(.+)'
    matches = re.findall(pattern, schedule_text, re.IGNORECASE)

    today = datetime.date.today()
    rows = []

    for start, end, task in matches:
        start_dt = datetime.datetime.combine(
            today, datetime.datetime.strptime(start, "%I:%M %p").time()
        )
        end_dt = datetime.datetime.combine(
            today, datetime.datetime.strptime(end, "%I:%M %p").time()
        )
        rows.append({"Start": start_dt, "End": end_dt, "Task": task})

    return pd.DataFrame(rows)


def visualize_schedule(schedule_text):
    df = parse_schedule_to_df(schedule_text)
    if df.empty:
        return px.scatter(title="No valid tasks found")

    fig = px.timeline(df, x_start="Start", x_end="End", y="Task", color="Task")
    fig.update_yaxes(autorange="reversed")
    return fig


# =========================
# MAIN FUNCTION
# =========================
def process_input(audio_file, text_input):
    try:
        transcription = ""

        if schedule_data["final_schedule"] and schedule_data["has_edited"]:
            return (
                "",
                "⚠ Schedule locked. No more edits allowed.",
                "N/A"
            )

        if audio_file:
            result = model.transcribe(audio_file)
            text = result["text"]
            transcription = text
        elif text_input:
            text = text_input
        else:
            return "No input provided", "", ""

        prompt = f"""
Create a clean daily schedule from this input:
{text}

Rules:
- Use 30 minute slots
- Format strictly as:
9:00 am - 9:30 am: Task
- Cover the whole day
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
)

        schedule_text = response.choices[0].message.content.strip()

        start, end = extract_dates(schedule_text)

        if schedule_data["final_schedule"] is None:
            schedule_data["final_schedule"] = schedule_text
            return transcription, schedule_text, f"{start} → {end}"

        schedule_data["final_schedule"] = schedule_text
        schedule_data["has_edited"] = True

        return (
            transcription,
            "✅ Schedule updated (final edit). Locked now.",
            f"{start} → {end}"
        )

    except Exception as e:
        return f"Error: {e}", "", ""


# =========================
# UI
# =========================
iface = gr.Interface(
    fn=process_input,
    inputs=[
        gr.Audio(type="filepath", label="Upload Audio (optional)"),
        gr.Textbox(label="Text Input (optional)")
    ],
    outputs=[
        gr.Textbox(label="Transcription"),
        gr.Textbox(label="Schedule"),
        gr.Textbox(label="Time Range")
    ],
    title="AI Daily Productivity Planner",
    description="Create a daily routine from voice or text. One edit only — then locked."
)

def reset_schedule(password):
    if password == "hackathon":
        schedule_data["final_schedule"] = None
        schedule_data["has_edited"] = False
        return "✅ Reset successful"
    return "❌ Wrong password"


reset_ui = gr.Interface(
    fn=reset_schedule,
    inputs=gr.Textbox(label="Developer Password"),
    outputs="text",
    title="Developer Reset"
)

with gr.Blocks() as demo:
    iface.render()

    gr.Markdown("## 📊 Visual Schedule")
    plot = gr.Plot()

    demo.load(
        fn=lambda: visualize_schedule(schedule_data["final_schedule"])
        if schedule_data["final_schedule"] else None,
        outputs=plot
    )

    reset_ui.render()

demo.launch()
