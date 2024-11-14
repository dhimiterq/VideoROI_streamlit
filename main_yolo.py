import cv2
import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
from ultralytics import YOLO
from PIL import Image
import base64
from io import BytesIO

# Initialize the YOLOv8 model
model = YOLO("yolov8n.pt")

def create_mask_from_closed_loop(frame_shape, drawing_data):
    """Creates a binary mask based on a closed loop from freehand drawing data."""
    mask = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.uint8)  # Create a blank mask

    if drawing_data and drawing_data["objects"]:
        for obj in drawing_data["objects"]:
            if obj["type"] == "path":  # Handle freehand path as a closed loop
                path = obj["path"]
                # Extract points and close the loop
                points = np.array([[p[1], p[2]] for p in path if len(p) > 2], dtype=np.int32)
                if len(points) > 2:
                    cv2.fillPoly(mask, [points], color=255)  # Fill the closed loop in the mask

    return mask

def convert_frame_to_base64(frame):
    """Convert a frame (image) to a base64 string."""
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_base64

def main():
    st.title("YOLOv8 Object Detection on Custom Freehand Mask")

    # Sidebar for canvas options
    st.sidebar.subheader("Draw your ROI with Freehand Tool")
    stroke_width = st.sidebar.slider("Stroke width", 1, 10, 3)
    drawing_mode = "freedraw"

    # Canvas for drawing ROI
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",  # Semi-transparent fill color
        stroke_width=stroke_width,
        stroke_color="rgba(255, 0, 0, 1)",  # Red outline for clarity
        background_color="rgba(0, 0, 0, 0)",
        update_streamlit=True,
        height=480,
        width=640,
        drawing_mode=drawing_mode,
        key="canvas",
    )

    # Start video capture
    video_feed = cv2.VideoCapture(0)
    if not video_feed.isOpened():
        st.error("Error: Could not open video feed.")
        return

    stframe1 = st.empty()
    stframe2 = st.empty()

    while True:
        ret, frame = video_feed.read()
        if not ret:
            break

        # Resize frame to match canvas size
        frame = cv2.resize(frame, (640, 480))

        # Generate mask from freehand closed loop
        mask = create_mask_from_closed_loop(frame.shape, canvas_result.json_data)
        
        # Apply mask to keep only the ROI inside the closed loop
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # Run YOLOv8 model on the masked frame
        results = model.predict(masked_frame)

        # Annotate the masked frame with bounding boxes and labels
        annotated_frame = results[0].plot()

        # Display both the original and annotated masked frames
        stframe1.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", caption="Original Video")
        stframe2.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", caption="Masked Video with YOLOv8 Detections")

    # Release the video capture object
    video_feed.release()

if __name__ == "__main__":
    main()
