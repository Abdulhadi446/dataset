import streamlit as st
import subprocess
import tempfile
import os
import glob

st.set_page_config(page_title="Piper TTS", page_icon="🎙️", layout="centered")

st.title("🎙️ Piper TTS")
st.caption("Local text-to-speech using Piper")

# Find available models
def find_models():
    patterns = [
        os.path.expanduser("~/**/*.onnx"),
        "/tmp/**/*.onnx",
        "./**/*.onnx",
        os.path.expanduser("~/Desktop/**/*.onnx"),
    ]
    models = []
    for pattern in patterns:
        models.extend(glob.glob(pattern, recursive=True))
    # Filter out .onnx.json files
    models = [m for m in models if not m.endswith(".json")]
    return models

models = find_models()

if not models:
    st.error("No .onnx model files found. Make sure your model is downloaded.")
    st.code("wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx")
    st.stop()

# Model selector
model_path = st.selectbox(
    "Voice model",
    models,
    format_func=lambda x: os.path.basename(x)
)

# Input method
input_method = st.radio("Input", ["Type / Paste text", "Upload .txt file"], horizontal=True)

text = ""
if input_method == "Type / Paste text":
    text = st.text_area("Text", placeholder="Paste your text here...", height=300)
else:
    uploaded = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded:
        text = uploaded.read().decode("utf-8")
        st.text_area("Preview", text, height=200, disabled=True)

# Speed slider
speed = st.slider("Speed", 0.5, 2.0, 1.0, 0.05)

# Generate
if st.button("🔊 Generate Audio", use_container_width=True, type="primary"):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Generating..."):
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    out_path = tmp.name

                result = subprocess.run(
                    ["piper", "--model", model_path, "--output_file", out_path, "--length_scale", str(1.0 / speed)],
                    input=text.encode("utf-8"),
                    capture_output=True,
                    timeout=300
                )

                if result.returncode != 0:
                    st.error(f"Piper error:\n{result.stderr.decode()}")
                elif os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                    with open(out_path, "rb") as f:
                        audio_bytes = f.read()
                    st.audio(audio_bytes, format="audio/wav")
                    st.download_button(
                        "⬇️ Download WAV",
                        data=audio_bytes,
                        file_name="output.wav",
                        mime="audio/wav",
                        use_container_width=True
                    )
                    os.unlink(out_path)
                else:
                    st.error("Output file empty or missing.")
            except FileNotFoundError:
                st.error("`piper` not found. Make sure it's installed: `pipx install piper-tts`")
            except subprocess.TimeoutExpired:
                st.error("Timed out — text might be too long, try splitting it.")
            except Exception as e:
                st.error(f"Error: {e}")
