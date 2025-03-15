from PIL import Image
import numpy as np
import tensorflow as tf

output_labels = ['vit K', 'vit C', 'vit D', 'vit B', 'vit A', 'vit E']

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='my_model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image):
    img = image.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0).astype(np.float32)

def predict_vitamin(image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    preprocessed_image = preprocess_image(image)
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], preprocessed_image)
    
    # Run inference
    interpreter.invoke()

    # Get the output and make the prediction
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data)
    predicted_label = output_labels[predicted_class]
    
    return predicted_label
