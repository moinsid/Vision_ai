
import threading

import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from detection import detect_objects


class VisionProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.counts = {}
        self.detections = []
        self.error = None

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        try:
            annotated, counts, detections = detect_objects(image)

            with self.lock:
                self.counts = counts
                self.detections = detections
                self.error = None

            return av.VideoFrame.from_ndarray(
                annotated, format="bgr24"
            )

        except Exception as exc:
            with self.lock:
                self.error = str(exc)
            return frame


st.set_page_config(
    page_title="Vision AI Live",
    page_icon="👁️",
    layout="wide",
)

st.title("👁️ Vision AI — Live Object Detection")
st.write("Start the camera to detect objects continuously.")

ctx = webrtc_streamer(
    key="vision-live",
    video_processor_factory=VisionProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    async_processing=True,
)

if ctx.video_processor:
    processor = ctx.video_processor

    with processor.lock:
        counts = processor.counts.copy()
        detections = list(processor.detections)
        error = processor.error

    if error:
        st.error(error)

    st.subheader("Live object counts")
    st.json(counts)

    st.subheader("Detection confidence")
    if detections:
        st.dataframe(detections, use_container_width=True)
    else:
        st.info("Waiting for objects to be detected...")
