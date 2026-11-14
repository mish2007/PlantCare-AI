from io import BytesIO
import json
import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError


st.set_page_config(
    page_title="PlantCare AI",
    page_icon="🌿",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #f5f8f4;
        color: #20352a;
    }

    .block-container {
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 1.8rem 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid #dce8dd;
        border-radius: 20px;
        background: linear-gradient(120deg, #e8f4e9, #f8fbf6);
    }

    .hero h1 {
        margin: 0;
        color: #1d5937;
        font-size: clamp(2rem, 5vw, 3rem);
    }

    .hero p {
        max-width: 680px;
        margin: 0.6rem 0 0;
        color: #52665a;
        font-size: 1.05rem;
        line-height: 1.6;
    }

    .section-card {
        height: 100%;
        padding: 1.2rem;
        border: 1px solid #e0e9e1;
        border-radius: 16px;
        background: #ffffff;
        box-shadow: 0 4px 16px rgba(32, 53, 42, 0.04);
    }

    .section-card h3 {
        margin: 0 0 0.7rem;
        color: #285d3d;
        font-size: 1.05rem;
    }

    .placeholder {
        color: #718076;
        line-height: 1.5;
    }

    div.stButton > button {
        min-height: 3rem;
        border: 0;
        border-radius: 10px;
        background: #287447;
        color: #ffffff;
        font-weight: 600;
    }

    div.stButton > button:hover {
        border: 0;
        background: #1f6039;
        color: #ffffff;
    }

    @media (max-width: 640px) {
        .block-container {
            padding: 1rem 1rem 2rem;
        }

        .hero {
            padding: 1.3rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🌿 PlantCare AI</h1>
        <p>
            Upload a tomato leaf image only; this app checks tomato leaves, not
            all plant species. Get a prediction from the trained model and
            practical care guidance.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.session_state.setdefault("prediction_history", [])
st.session_state.setdefault("prediction_result", None)

APP_DIRECTORY = Path(__file__).resolve().parent
DISEASE_GUIDANCE = {
    "Tomato___Bacterial_spot": (
        "Bacterial spot is a bacterial disease that causes dark, water-soaked "
        "spots on leaves and can lead to yellowing.",
        "Avoid wetting foliage, remove affected leaves, sanitize tools, and "
        "avoid working with plants when they are wet.",
    ),
    "Tomato___Early_blight": (
        "Early blight commonly causes brown spots with concentric rings, often "
        "starting on older leaves.",
        "Remove affected lower leaves, keep foliage dry, mulch to reduce soil "
        "splash, and rotate crops where possible.",
    ),
    "Tomato___Late_blight": (
        "Late blight can cause rapidly expanding dark lesions and may spread "
        "quickly in cool, damp conditions.",
        "Remove affected plant material, improve airflow, avoid overhead "
        "watering, and seek local extension advice if it is spreading.",
    ),
    "Tomato___Leaf_Mold": (
        "Leaf mold often develops in humid conditions and may cause pale "
        "patches above leaves with mold growth underneath.",
        "Increase spacing and ventilation, reduce humidity, water at the soil "
        "level, and remove heavily affected leaves.",
    ),
    "Tomato___Septoria_leaf_spot": (
        "Septoria leaf spot typically causes many small, circular spots with "
        "light centers and dark margins.",
        "Remove affected lower leaves, avoid overhead watering, mulch around "
        "plants, and rotate crops.",
    ),
    "Tomato___healthy": (
        "The model found no signs of the six conditions it was trained to "
        "recognize. Continue monitoring the plant.",
        "Water consistently at the soil level, provide good airflow and "
        "sunlight, and inspect leaves regularly.",
    ),
}


def _configured_path(environment_variable: str, default_relative_path: str) -> Path:
    configured_path = os.environ.get(environment_variable)
    path = Path(configured_path).expanduser() if configured_path else Path(
        default_relative_path
    )
    if not path.is_absolute():
        path = APP_DIRECTORY / path
    return path.resolve()


@st.cache_resource
def _load_model_and_class_names(model_path: str, class_names_path: str):
    class_names_file = Path(class_names_path)
    model_file = Path(model_path)

    with class_names_file.open("r", encoding="utf-8") as file:
        class_names = json.load(file)
    if (
        not isinstance(class_names, list)
        or not all(isinstance(name, str) for name in class_names)
        or len(class_names) != len(DISEASE_GUIDANCE)
        or set(class_names) != set(DISEASE_GUIDANCE)
    ):
        raise ValueError(
            "class_names.json must be a JSON array containing the six expected "
            "class names."
        )

    import tensorflow as tf

    model = tf.keras.models.load_model(model_file)
    return model, class_names


st.header("Analyze a leaf")
upload_column, preview_column = st.columns(2)

with upload_column:
    uploaded_file = st.file_uploader(
        "Choose a tomato leaf image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, and PNG.",
    )
    st.warning(
        "Important: This version of PlantCare AI is trained only to identify "
        "six tomato leaf categories. Please upload tomato leaf images only. "
        "Images of other plants may receive incorrect predictions."
    )

    supported_image_type = (
        uploaded_file is not None
        and uploaded_file.name.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    valid_image = False
    if supported_image_type:
        image_bytes = uploaded_file.getvalue()
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                image.verify()
            with Image.open(BytesIO(image_bytes)) as image:
                image.load()
            valid_image = True
        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            Image.DecompressionBombError,
        ):
            st.error(
                "This is not a valid image. Please upload a proper JPG, JPEG, "
                "or PNG image."
            )

    if uploaded_file is None:
        st.info("Upload a JPG, JPEG, or PNG image to get started.")
    elif not supported_image_type:
        st.error("Unsupported file type. Please upload a JPG, JPEG, or PNG image.")

    predict_clicked = st.button(
        "Predict Disease",
        type="primary",
        use_container_width=True,
    )

    if predict_clicked:
        if valid_image:
            st.session_state["prediction_result"] = None
            model_path = _configured_path(
                "PLANTCARE_MODEL_PATH", "models/plantcare_model.keras"
            )
            class_names_path = _configured_path(
                "PLANTCARE_CLASS_NAMES_PATH", "models/class_names.json"
            )
            try:
                model, class_names = _load_model_and_class_names(
                    str(model_path), str(class_names_path)
                )
            except Exception as error:
                st.error(
                    "Could not load the trained model or class names. Confirm "
                    "that PLANTCARE_MODEL_PATH and PLANTCARE_CLASS_NAMES_PATH "
                    "point to accessible files, and that TensorFlow is allowed "
                    f"to run in this environment. Details: {error}"
                )
            else:
                try:
                    image = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")
                    image = image.resize((224, 224))
                    image_array = np.asarray(image, dtype=np.float32)
                    image_array = np.expand_dims(image_array, axis=0)

                    probabilities = np.asarray(
                        model.predict(image_array, verbose=0)
                    )
                    if (
                        probabilities.ndim != 2
                        or probabilities.shape[0] != 1
                        or probabilities.shape[1] != len(class_names)
                    ):
                        raise ValueError(
                            "The model output does not match the loaded class names."
                        )

                    predicted_index = int(np.argmax(probabilities[0]))
                    disease_name = class_names[predicted_index]
                    confidence = float(probabilities[0][predicted_index]) * 100
                    result = {
                        "disease": disease_name,
                        "confidence": confidence,
                        "image": uploaded_file.name,
                    }
                    st.session_state["prediction_result"] = result
                    st.session_state["prediction_history"].append(result.copy())
                except Exception as error:
                    st.error(f"Could not analyze this image. Details: {error}")
        else:
            st.warning("Please upload a valid image before requesting a prediction.")

with preview_column:
    st.subheader("Image preview")
    if valid_image:
        st.image(uploaded_file, caption=uploaded_file.name, width="stretch")
    else:
        st.markdown(
            '<div class="section-card placeholder">'
            "Your uploaded leaf image will appear here."
            "</div>",
            unsafe_allow_html=True,
        )

st.markdown("---")
st.header("Prediction result")
st.caption("Predictions are made using the trained tomato-leaf model.")

prediction_result = st.session_state["prediction_result"]
if prediction_result:
    disease_name = prediction_result["disease"]
    disease_label = disease_name.replace("___", " — ").replace("_", " ")
    disease_information, care_suggestions = DISEASE_GUIDANCE[disease_name]
    result_sections = [
        ("Predicted disease", disease_label),
        ("Confidence score", f"{prediction_result['confidence']:.2f}%"),
        ("Disease information", disease_information),
        ("Suggested care", care_suggestions),
    ]
else:
    result_sections = [
        ("Predicted disease", "No prediction yet"),
        ("Confidence score", "Not available"),
        ("Disease information", "Information will appear with a prediction."),
        ("Suggested care", "Care guidance will appear with a prediction."),
    ]

result_columns = st.columns(4)

for column, (heading, placeholder) in zip(result_columns, result_sections):
    with column:
        st.markdown(
            f"""
            <div class="section-card">
                <h3>{heading}</h3>
                <div class="placeholder">{placeholder}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")
st.header("Prediction history")

if st.session_state["prediction_history"]:
    for item in reversed(st.session_state["prediction_history"]):
        disease_label = item["disease"].replace("___", " — ").replace("_", " ")
        st.write(
            f"{item['image']} — {disease_label} "
            f"({item['confidence']:.2f}%)"
        )
else:
    st.info("No predictions yet. Your completed analyses will appear here.")
