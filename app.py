import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from PIL import Image

def sort_array_func(val):
    return val[3]

st.set_page_config(page_title="")

hide_st_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            # header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("")

choice = st.selectbox("select", ["Upload image", "Upload video"])
conf = st.number_input("conf", 0.2)

if choice == "Upload image":
    
    image_data = st.file_uploader("Upload the Image")
    img_summit_button = st.button("Predict")
    
    if img_summit_button:
        
        model = YOLO('CarPark2.pt')   
    
        image = Image.open(image_data)
        image.save("input_data_image.png")
        frame = cv2.imread("input_data_image.png")
        frame_without_condition = frame
                
        results = model.predict(source=frame, iou=0.7, conf=conf)
        plot_show =  results[0].plot()

        st.image(plot_show)

elif choice == "Upload video":

    video_data = st.file_uploader("Upload the Video", type=["mp4", "mov", "avi", "asf", "m4v"])
    video_summit_button = st.button("Predict")

    stop_video = st.checkbox("Stop Video Processing")

    if video_summit_button:
        
        model = YOLO('CarPark2.pt')   
        
        if video_data is not None:
            # Save the uploaded video to a temporary file
            temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_video_path.write(video_data.read())
            
            cap = cv2.VideoCapture(temp_video_path.name)

            # Check if video is successfully opened
            if not cap.isOpened():
                st.error("Error opening video file. Please try again.")
                st.stop()

            # Loop through the video frames
            while cap.isOpened() and not stop_video:
                # Read a frame from the video
                success, frame = cap.read()

                if success:
                    # Run YOLOv8 tracking on the frame
                    results = model.predict(source=frame, iou=0.7, conf=conf)
                    plot_show = results[0].plot()

                    # Display the annotated frame
                    st.image(plot_show, channels='BGR', use_column_width=True)

            # Release the video capture object
            cap.release()
