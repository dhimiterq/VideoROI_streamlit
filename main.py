import cv2
import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas

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

def main():
    st.title("Video Feed with Custom Freehand Mask (Closed Loop)")

    # Sidebar for canvas options
    st.sidebar.subheader("Draw your ROI with Freehand Tool")
    stroke_width = st.sidebar.slider("Stroke width", 1, 10, 3)
    drawing_mode = "freedraw"  # Set to freedraw mode

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
        
        # Invert the mask to keep only the ROI inside the closed loop
        inverted_mask = cv2.bitwise_not(mask)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)  # ROI
        background_frame = cv2.bitwise_and(frame, frame, mask=inverted_mask)  # Outside ROI set to black

        # Display the masked video with only the ROI visible
        masked_frame_rgb = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)

        # Display both frames side by side
        stframe1.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", caption="Original Video")
        stframe2.image(masked_frame_rgb, channels="RGB", caption="Masked Video with Freehand ROI")

    # Release the video capture object
    video_feed.release()

if __name__ == "__main__":
    main()
