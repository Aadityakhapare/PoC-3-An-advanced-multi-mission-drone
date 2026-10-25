
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
# CUSTOM ANNOTATION COLORS
# OpenCV uses BGR color format
# --------------------------------------------------
CLASS_COLORS = {
    0: (0, 200, 0),      # Helmet: GREEN
    1: (255, 0, 0),      # Person: BLUE
    2: (0, 0, 230)       # Safety Vest: RED
}

# --------------------------------------------------
# CUSTOM BOXES AND LABELS
# --------------------------------------------------
def draw_ppe_detections(result):
    image = result.orig_img.copy()
    height, width = image.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(width, height) / 1200.0)
    text_thickness = 1
    box_thickness = 2
    padding = 4

    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            class_name = str(
                model.names.get(class_id, class_id)
            )

            color = CLASS_COLORS.get(
                class_id, (255, 255, 255)
            )

            label = f"{class_name} {confidence:.2f}"

            # Keep bounding box coordinates inside the frame
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            detections.append(
                (x1, y1, x2, y2, label, color)
            )

    # Draw all bounding boxes first
    for x1, y1, x2, y2, label, color in detections:
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            box_thickness
        )

    # Draw colored label backgrounds and white text
    for x1, y1, x2, y2, label, color in detections:
        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            text_thickness
        )

        label_w = text_w + 2 * padding
        label_h = text_h + baseline + 2 * padding

        # Keep labels inside the video frame horizontally
        label_x = max(
            0,
            min(x1, max(0, width - label_w))
        )

        # Prefer placing the label above the box
        label_top = y1 - label_h - 2

        # If there is no space above, place it below
        if label_top < 0:
            label_top = y2 + 2

        # If there is no space below, place it near the box
        if label_top + label_h >= height:
            label_top = max(0, y1)

        label_top = max(0, min(label_top, height - 1))
        label_bottom = min(height - 1, label_top + label_h)
        label_right = min(width - 1, label_x + label_w)

        # Matching color label background
        cv2.rectangle(
            image,
            (label_x, label_top),
            (label_right, label_bottom),
            color,
            thickness=-1
        )

        # White label text
        text_y = min(
            height - baseline - 1,
            label_top + padding + text_h
        )

        cv2.putText(
            image,
            label,
            (label_x + padding, text_y),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA
        )

    return image

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
                conf=0.30,
                imgsz=384,
                iou=0.50,
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

            # Custom colored bounding boxes and label backgrounds
            annotated_frame = draw_ppe_detections(result)

            writer.write(annotated_frame)

            # Progress
            progress = min(
                frame_number / max(total_frames, 1),
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
