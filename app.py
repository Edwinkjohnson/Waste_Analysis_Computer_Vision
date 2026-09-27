# Import Streamlit for creating the web application
import streamlit as st

# Import YOLO models from Ultralytics
from ultralytics import YOLO

# Import PIL for opening uploaded images
from PIL import Image

# Import NumPy for converting images into arrays
import numpy as np

# Import Counter for counting waste categories
from collections import Counter


# =========================================================
# PAGE CONFIGURATION
# =========================================================

# Configure the Streamlit page
st.set_page_config(
    page_title="Waste Classification & Segmentation",
    page_icon="♻️",
    layout="wide"
)


# =========================================================
# CLASS NAMES
# =========================================================

# Class names used by the segmentation model
SEGMENTATION_CLASSES = [
    "glass",
    "household_waste",
    "metal",
    "paper_cardboard",
    "plastic"
]


# Class names used by the classification model
CLASSIFICATION_CLASSES = [
    "glass",
    "household_waste",
    "metal",
    "paper_cardboard",
    "plastic"
]


# =========================================================
# LOAD MODELS
# =========================================================

# Load the waste-pile segmentation model
@st.cache_resource
def load_segmentation_model():

    # best.pt is the YOLO segmentation model
    model = YOLO("best.pt")

    # Return the loaded model
    return model


# Load the single-item classification model
@st.cache_resource
def load_classification_model():

    # single_item_classifier.pt is our new classifier
    model = YOLO("single_item_classifier.pt")

    # Return the loaded model
    return model


# Load both models
segmentation_model = load_segmentation_model()
classification_model = load_classification_model()


# =========================================================
# TITLE
# =========================================================

# Display the application title
st.title("♻️ Waste Classification & Segmentation")

# Display a short description
st.write(
    "Classify individual waste items or segment and count "
    "multiple waste items in a pile."
)


# =========================================================
# SIDEBAR
# =========================================================

# Create the sidebar
st.sidebar.header("⚙️ Settings")


# Allow the user to select the prediction mode
mode = st.sidebar.radio(
    "Select Prediction Mode",
    [
        "Single Item",
        "Waste Pile"
    ]
)


# =========================================================
# SINGLE ITEM MODE
# =========================================================

if mode == "Single Item":

    # Display section heading
    st.header("📷 Single Waste Item")

    # Explain what the user should upload
    st.write(
        "Upload an image containing one waste item."
    )

    # Examples of suitable inputs
    st.info(
        "Examples: plastic bottle, metal can, glass bottle, "
        "paper/cardboard, or household waste."
    )

    # Allow the user to upload one image
    uploaded_file = st.file_uploader(
        "Upload a single waste item",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        key="single_item_uploader"
    )


    # Continue only if an image has been uploaded
    if uploaded_file is not None:

        # Open the uploaded image
        image = Image.open(
            uploaded_file
        ).convert("RGB")


        # Display the original image
        st.subheader("Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )


        # Create the prediction button
        if st.button(
            "🔍 Classify Waste Item",
            type="primary"
        ):

            # Convert the PIL image into a NumPy array
            image_array = np.array(image)


            # Run the classification model
            results = classification_model.predict(
                source=image_array,

                # Classification model was trained with 224x224 images
                imgsz=224,

                # CPU is used if no GPU is available
                device="cpu",

                # Do not print YOLO logs in Streamlit
                verbose=False
            )


            # Get the first prediction result
            result = results[0]


            # Get the class ID with the highest probability
            predicted_class_id = result.probs.top1


            # Get the confidence of the highest prediction
            predicted_confidence = (
                result.probs.top1conf.item()
            )


            # Convert class ID into class name
            predicted_class = (
                CLASSIFICATION_CLASSES[
                    predicted_class_id
                ]
            )


            # =================================================
            # DISPLAY SINGLE ITEM RESULT
            # =================================================

            st.subheader("Prediction")


            # Create three columns
            col1, col2, col3 = st.columns(3)


            # Display predicted category
            with col1:

                st.metric(
                    "Waste Category",
                    predicted_class
                )


            # Display confidence
            with col2:

                st.metric(
                    "Confidence",
                    f"{predicted_confidence * 100:.2f}%"
                )


            # Display model type
            with col3:

                st.metric(
                    "Model",
                    "Single-item classifier"
                )


            # =================================================
            # TOP PREDICTIONS
            # =================================================

            st.subheader("Prediction Probabilities")


            # Get the top 5 predicted class IDs
            top5_class_ids = result.probs.top5


            # Get the corresponding confidence values
            top5_confidences = (
                result.probs.top5conf.cpu().numpy()
            )


            # Display all five class probabilities
            for class_id, confidence_value in zip(
                top5_class_ids,
                top5_confidences
            ):

                # Convert ID to class name
                class_name = (
                    CLASSIFICATION_CLASSES[
                        class_id
                    ]
                )


                # Display class and confidence
                st.write(
                    f"**{class_name}** — "
                    f"{confidence_value * 100:.2f}%"
                )


# =========================================================
# WASTE PILE MODE
# =========================================================

else:

    # Display section heading
    st.header("🗑️ Waste Pile")

    # Explain the pile functionality
    st.write(
        "Upload an image containing multiple waste items. "
        "The segmentation model will detect, segment, "
        "classify, and count the objects."
    )


    # Confidence threshold for segmentation
    confidence = st.sidebar.slider(
        "Segmentation Confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05
    )


    # Select YOLO image size
    image_size = st.sidebar.selectbox(
        "Segmentation Image Size",
        [640, 768, 960],
        index=0
    )


    # Allow multiple images
    uploaded_files = st.file_uploader(
        "Upload waste pile image(s)",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        accept_multiple_files=True,
        key="pile_uploader"
    )


    # Continue if images were uploaded
    if uploaded_files:

        # Process each uploaded image
        for uploaded_file in uploaded_files:

            # Display filename
            st.subheader(
                f"📷 {uploaded_file.name}"
            )


            # Open image
            image = Image.open(
                uploaded_file
            ).convert("RGB")


            # Convert image to NumPy array
            image_array = np.array(image)


            # Run the segmentation model
            results = segmentation_model.predict(
                source=image_array,

                # Confidence threshold
                conf=confidence,

                # Image size
                imgsz=image_size,

                # Use CPU
                device="cpu",

                # Hide YOLO logs
                verbose=False
            )


            # Get first prediction
            result = results[0]


            # =================================================
            # DISPLAY ORIGINAL + ANNOTATED IMAGE
            # =================================================

            # Create two columns
            col1, col2 = st.columns(2)


            # Original image
            with col1:

                st.write("### Original")

                st.image(
                    image,
                    use_container_width=True
                )


            # Annotated image
            with col2:

                st.write("### Segmentation")

                # Create image with masks, boxes and labels
                annotated_image = result.plot()


                # YOLO returns BGR format
                # Convert BGR to RGB for Streamlit
                annotated_image = (
                    annotated_image[:, :, ::-1]
                )


                # Display annotated image
                st.image(
                    annotated_image,
                    use_container_width=True
                )


            # =================================================
            # COUNT DETECTED OBJECTS
            # =================================================

            if (
                result.boxes is not None
                and len(result.boxes) > 0
            ):

                # Get detected class IDs
                class_ids = (
                    result.boxes.cls
                    .cpu()
                    .numpy()
                    .astype(int)
                )


                # Convert IDs into class names
                detected_classes = [
                    SEGMENTATION_CLASSES[class_id]
                    for class_id in class_ids
                ]


                # Count each category
                counts = Counter(
                    detected_classes
                )


                # Display total
                st.subheader(
                    f"Total detected items: "
                    f"{len(detected_classes)}"
                )


                # Create columns for category counts
                count_columns = st.columns(
                    len(counts)
                )


                # Display each category count
                for column, (
                    class_name,
                    count
                ) in zip(
                    count_columns,
                    counts.items()
                ):

                    with column:

                        st.metric(
                            class_name,
                            count
                        )


            else:

                # Display warning if nothing was detected
                st.warning(
                    "No waste items were detected. "
                    "Try lowering the confidence threshold."
                )