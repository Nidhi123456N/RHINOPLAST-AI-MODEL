import streamlit as st
from pipeline_handler import load_pipeline, generate_image
from PIL import Image

# Title of the app
st.title("Rhinoplasty AI Tool")

# Load the pipeline
st.sidebar.text("Loading pipeline...")
pipe = None
try:
    pipe = load_pipeline()
    st.sidebar.success("Pipeline loaded!")
except Exception as pipeline_error:
    st.sidebar.error(f"Error loading pipeline: {pipeline_error}")

# Function to handle .jfif files
def handle_jfif(file):
    try:
        jfif_img = Image.open(file).convert("RGB")
        return jfif_img
    except Exception as jfif_error:
        st.error(f"Error processing .jfif file: {jfif_error}")
        return None

# File uploader for images
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg", "jfif"])

if uploaded_file:
    try:
        # Detect the file type and process it accordingly
        if uploaded_file.name.lower().endswith(".jfif"):
            processed_img = handle_jfif(uploaded_file)
        else:
            processed_img = Image.open(uploaded_file).convert("RGB")

        # Ensure the image is resized for pipeline compatibility
        img_resized = processed_img.resize((512, 512))
        st.image(img_resized, caption="Resized Image (512x512)", use_container_width=True)

        # Dropdown menu for selecting rhinoplasty effects
        prompts = [
            "A photorealistic side profile of a person with a narrower and straighter nose",
            "A front portrait of a person with a small, sharp nasal tip",
            "A side profile of a person with a well-defined nasal bridge"
        ]
        selected_prompt = st.selectbox("Choose a rhinoplasty effect:", prompts)

        # Generate button
        if st.button("Generate"):
            with st.spinner("Generating the modified image..."):
                result_image = generate_image(pipe, selected_prompt, img_resized)

                # Display and save the result
                st.image(result_image, caption="Modified Image", use_container_width=True)
                result_image.save("output.png")
                st.success("Modified image saved as 'output.png'.")

    except Exception as app_error:
        st.error(f"An error occurred: {app_error}")




