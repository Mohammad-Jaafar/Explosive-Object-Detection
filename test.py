import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Image properties
IMG_SIZE = (224, 224)
class_names = ['Exp', 'Safe']  # Model classes

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="models/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(img_path):
    """
    Load an image from path, convert to RGB,
    resize to target size, and prepare for TFLite input.
    """
    img = Image.open(img_path).convert('RGB')
    img = img.resize(IMG_SIZE)
    img = np.array(img, dtype=np.uint8)  
    return np.expand_dims(img, axis=0)

def predict_tflite(img_path):
    """
    Run inference using TFLite model and display prediction with confidence.
    """
    input_data = preprocess_image(img_path)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = output_data.astype(np.float32) / 255.0  # Normalize
    predicted_class = class_names[np.argmax(output_data)]
    confidence = np.max(output_data)

    # Show image with prediction
    img = Image.open(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Predicted: {predicted_class} ({confidence*100:.2f}%)')
    plt.show()

# Run inference on a sample image
predict_tflite('image.jpg')
