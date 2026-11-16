
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path

# 1. Configure the Streamlit page
st.set_page_config(
    page_title="Rice Leaf Disease Detection",
    page_icon="🌾",
    layout="centered"
)

# 2. Define class names and model settings
CLASS_NAMES = [
    "Bacterial leaf blight",
    "Brown spot",
    "Leaf smut"
]

IMG_SIZE = (128, 128)
MODEL_PATH = "rice_leaf_disease_cnn.keras"


# 3. Load the trained CNN model
@st.cache_resource
def load_cnn_model():

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(MODEL_PATH)


# 4. Preprocess uploaded image
def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(IMG_SIZE)

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# 5. Display application title
st.title("🌾 Rice Leaf Disease Detection")

st.write(
    "Upload a rice leaf image to identify its disease "
    "using a Convolutional Neural Network (CNN)."
)

st.divider()


# 6. Upload image
uploaded_file = st.file_uploader(
    "Upload Rice Leaf Image",
    type=["jpg", "jpeg", "png"]
)


# 7. Display image and predict
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Rice Leaf Image",
        width=350
    )

    if st.button("Predict Disease", type="primary"):

        try:
            with st.spinner("Analyzing image..."):

                model = load_cnn_model()

                processed_image = preprocess_image(image)

                prediction = model.predict(
                    processed_image,
                    verbose=0
                )

                predicted_index = int(
                    np.argmax(prediction[0])
                )

                predicted_disease = CLASS_NAMES[
                    predicted_index
                ]

                confidence = float(
                    prediction[0][predicted_index] * 100
                )

            # 8. Display prediction result
            st.success("Prediction Completed!")

            st.subheader("Prediction Result")

            st.write(
                f"**Predicted Disease:** {predicted_disease}"
            )

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

            # 9. Display all class probabilities
            st.subheader("Class Probabilities")

            for i, disease in enumerate(CLASS_NAMES):

                probability = float(prediction[0][i])

                st.write(
                    f"{disease}: {probability * 100:.2f}%"
                )

                st.progress(probability)

        except Exception as e:

            st.error(f"Prediction failed: {e}")


st.divider()

st.caption(
    "Developed using Python, TensorFlow, Keras and Streamlit."
)
