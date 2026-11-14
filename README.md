
# PlantCare AI – Tomato Leaf Disease Detection System

PlantCare AI is a Neural Network based application that detects diseases in tomato leaves using uploaded images.

The project uses a trained deep learning model and a Streamlit user interface to classify tomato leaf images into six categories.

## Features

- Upload tomato leaf images.
- Preview uploaded images.
- Predict the tomato leaf category using a trained neural network.
- Display prediction confidence.
- Show disease information and care suggestions.
- Maintain prediction history during the application session.
- Validate uploaded images and display friendly error messages.

## Supported Categories

1. Tomato Bacterial Spot
2. Tomato Early Blight
3. Tomato Healthy
4. Tomato Late Blight
5. Tomato Leaf Mold
6. Tomato Septoria Leaf Spot

## Model Information

- Model: MobileNetV2 Transfer Learning
- Input image size: 224 × 224
- Output classes: 6
- Training epochs: 12
- Best validation accuracy: 90.18%
- Test accuracy: 89.38%

## Technologies Used

- Python
- TensorFlow / Keras
- MobileNetV2
- Streamlit
- NumPy
- Pillow
- Google Colab

## Project Structure

```text
PlantCare-AI/
│
├── app.py
├── prepare_dataset.py
├── requirements.txt
├── README.md
│
└── models/
    ├── plantcare_model.keras
    └── class_names.json
```

## Installation and Usage

1. Clone or download this repository.
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app.py
```

4. Open the local Streamlit URL in your browser.
5. Upload a tomato leaf image and click the prediction button.

## Important Limitation

This version is trained only on six tomato leaf categories. It is not designed to identify diseases in other plant species.

Images of non-tomato plants may receive incorrect predictions.

The model's confidence score represents its prediction among the six supported categories and should not be treated as a guaranteed diagnosis.

## Model Evaluation

The model achieved 89.38% accuracy on the held-out test dataset.

This is an academic project developed as part of the Neural Networks course.

## Author

Trishna Mishra
