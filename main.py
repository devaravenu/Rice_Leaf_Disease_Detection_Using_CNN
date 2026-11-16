
import numpy as np
import tensorflow as tf
from PIL import Image
import argparse
from pathlib import Path

# 1. Define class names in the same order as model training
CLASS_NAMES = [
    "Bacterial leaf blight",
    "Brown spot",
    "Leaf smut"
]

# 2. Define image size and model path
IMG_SIZE = (128, 128)
MODEL_PATH = "rice_leaf_disease_cnn.keras"


# 3. Load the trained CNN model
def load_model():
    model_path = Path(MODEL_PATH)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(model_path)
    return model


# 4. Preprocess the input image
def preprocess_image(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(image_path).convert("RGB")

    image = image.resize(IMG_SIZE)

    image_array = np.array(image, dtype=np.float32)

    image_array = image_array / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# 5. Predict the disease
def predict_disease(image_path):

    model = load_model()

    processed_image = preprocess_image(image_path)

    prediction = model.predict(
        processed_image,
        verbose=0
    )

    predicted_index = int(np.argmax(prediction[0]))

    predicted_disease = CLASS_NAMES[predicted_index]

    confidence = float(
        prediction[0][predicted_index] * 100
    )

    # 6. Display prediction result
    print("\n----- Rice Leaf Disease Prediction -----")

    print("Predicted Disease:", predicted_disease)

    print(f"Confidence: {confidence:.2f}%")

    print("\nClass Probabilities:")

    for i, disease in enumerate(CLASS_NAMES):
        probability = float(prediction[0][i] * 100)

        print(f"{disease}: {probability:.2f}%")

    print("----------------------------------------")


# 7. Main execution
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Rice Leaf Disease Prediction"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path of the rice leaf image"
    )

    args = parser.parse_args()

    predict_disease(args.image)
