import streamlit as st
from ultralytics import YOLO
import tempfile
import os
import cv2
import subprocess

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="PPE Kit Detection System",
    page_icon="🦺",
    layout="wide"
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("🦺 PPE Kit Detection System")
st.caption("YOLOv8-based detection of Person, Helmet and Safety Vest")

st.divider()

# --------------------------------------------------
# MODEL
# --------------------------------------------------
MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    st.error("❌ best.pt not found. Keep best.pt in the same folder as app.py.")
    st.stop()


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()

# --------------------------------------------------
# VIDEO UPLOAD
# --------------------------------------------------
st.subheader("📹 Upload PPE Video")

uploaded_file = st.file_uploader(
    "Choose a video file",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_file is not None:

    st.success(f"Uploaded: {uploaded_file.name}")

    # --------------------------------------------------
    # DETECTION BUTTON
    # --------------------------------------------------
    if st.button("🚀 Detect PPE", type="primary"):

        # Save uploaded video
        suffix = os.path.splitext(uploaded_file.name)[1]

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_video:

            temp_video.write(uploaded_file.read())
            input_path = temp_video.name

        # --------------------------------------------------
        # OUTPUT DIRECTORY
        # --------------------------------------------------
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        raw_output = os.path.join(
            output_dir,
            "PPE_detection_raw.mp4"
        )

        final_output = os.path.join(
            output_dir,
            "PPE_detection_result.mp4"
        )

        # --------------------------------------------------
        # VIDEO INFORMATION
        # --------------------------------------------------
        cap = cv2.VideoCapture(input_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.release()

        if fps <= 0:
            fps = 25

        # Resize output for faster processing
        output_width = 640
        output_height = int(
            height * (output_width / width)
        )

        output_height -= output_height % 2

        # --------------------------------------------------
        # VIDEO WRITER
        # --------------------------------------------------
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            raw_output,
            fourcc,
            fps,
            (output_width, output_height)
        )

        # --------------------------------------------------
        # PROCESSING UI
        # --------------------------------------------------
        st.subheader("🔄 Processing Video")

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Maximum detections seen in a single frame
        max_persons = 0
        max_helmets = 0
        max_vests = 0

        frame_number = 0

        # --------------------------------------------------
        # PROCESS VIDEO FRAME BY FRAME
        # --------------------------------------------------
        cap = cv2.VideoCapture(input_path)

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            # Resize frame
            frame_resized = cv2.resize(
                frame,
                (output_width, output_height)
            )

            # YOLO inference
            results = model.predict(
                frame_resized,
                conf=0.25,
                imgsz=384,
                device="cpu",
                verbose=False
            )

            result = results[0]

            # Detection counters
            persons = 0
            helmets = 0
            vests = 0

            if result.boxes is not None:

                for cls in result.boxes.cls:

                    class_id = int(cls)

                    if class_id == 0:
                        helmets += 1

                    elif class_id == 1:
                        persons += 1

                    elif class_id == 2:
                        vests += 1

            max_persons = max(max_persons, persons)
            max_helmets = max(max_helmets, helmets)
            max_vests = max(max_vests, vests)

            # Draw bounding boxes
            annotated_frame = result.plot()

            annotated_frame = cv2.resize(
                annotated_frame,
                (output_width, output_height)
            )

            writer.write(annotated_frame)

            # Progress
            progress = min(
                frame_number / total_frames,
                1.0
            )

            progress_bar.progress(progress)

            status_text.write(
                f"Processing frame {frame_number}/{total_frames}"
            )

        cap.release()
        writer.release()

        # --------------------------------------------------
        # CONVERT TO H.264
        # --------------------------------------------------
        status_text.write(
            "🔄 Converting video for browser playback..."
        )

        ffmpeg_command = [
            "ffmpeg",
            "-y",
            "-i",
            raw_output,
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            final_output
        ]

        try:

            subprocess.run(
                ffmpeg_command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

        except FileNotFoundError:

            st.error(
                "❌ FFmpeg is not installed on your computer. "
                "Please install FFmpeg and restart the application."
            )

            st.stop()

        except subprocess.CalledProcessError as e:

            st.error("❌ Video conversion failed.")
            st.text(e.stderr.decode(errors="ignore"))
            st.stop()

        # --------------------------------------------------
        # FINISHED
        # --------------------------------------------------
        progress_bar.progress(1.0)

        status_text.success(
            "✅ PPE detection completed!"
        )

        # Remove temporary input
        try:
            os.remove(input_path)
        except:
            pass

        # --------------------------------------------------
        # RESULTS
        # --------------------------------------------------
        st.divider()

        st.subheader("📊 Detection Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "👤 Persons",
                max_persons
            )

        with col2:
            st.metric(
                "⛑️ Helmets",
                max_helmets
            )

        with col3:
            st.metric(
                "🦺 Safety Vests",
                max_vests
            )

        st.divider()

        # --------------------------------------------------
        # FINAL VIDEO
        # --------------------------------------------------
        st.subheader("🎥 Annotated Video")

        if os.path.exists(final_output):

            with open(final_output, "rb") as video_file:
                video_bytes = video_file.read()

            st.video(video_bytes)

            st.download_button(
                label="⬇️ Download Detection Result",
                data=video_bytes,
                file_name="PPE_detection_result.mp4",
                mime="video/mp4"
            )

        else:
            st.error(
                "❌ Final output video was not created."
            )

        st.divider()

        st.info(
            "The YOLOv8 model detects three classes: "
            "Person, Helmet and Safety Vest."
        )