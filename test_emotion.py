# test_emotion.py
# Ensure to install the required packages by running:
# pip install deepface tf-keras

from deepface import DeepFace

def detect_emotion(image_path):
    analysis = DeepFace.analyze(image_path, actions=['emotion'])
    return analysis

if __name__ == "__main__":
    image_path = "path_to_your_image.jpg"  # Update with your image path
    result = detect_emotion(image_path)
    print(result)