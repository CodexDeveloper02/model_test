from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image
import os

app = Flask(__name__)
model = tf.keras.models.load_model('D:/Notes semester-8/FYP-2_updated/Project_Backend/Model_File/DN21_model.h5')


def preprocess_image(image):
    try:
        # Resize the image to (224, 224)
        image = image.resize((224, 224)) 
        image = np.array(image)
        image = image / 255
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        raise ValueError("Error in preprocessing image: {}".format(str(e)))

class_name = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

def predict(image):
    try:
        preprocessed_image = preprocess_image(image)
        predictions = model.predict(preprocessed_image)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_name[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        return predicted_class, confidence
    except Exception as e:
        raise ValueError("Error in predicting: {}".format(str(e)))

@app.route('/predict', methods=['GET', 'POST'])
def predict_tumor():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        try:
            print('MRI', file.filename)
            if file.filename.lower().endswith('.jpg'):
                image = Image.open(file)
                predicted_class, confidence = predict(image)
                return jsonify({'predicted_class': predicted_class, 'confidence': confidence})
            else:
                return jsonify({'error': 'Invalid file format. Must be jpg'})
        except Exception as e:
            return jsonify({'error': str(e)})

    elif request.method == 'GET':
        return jsonify({'message': 'Send a POST request with an image file to predict brain tumor.'})

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)

# import logging
# from flask import Flask, request, jsonify
# import tensorflow as tf
# import numpy as np
# from werkzeug.utils import secure_filename
# from PIL import Image
# import os

# # Setup logging
# logging.basicConfig(level=logging.INFO)

# app = Flask(__name__)
# model = tf.keras.models.load_model("D:/Notes semester-8/FYP-2_updated/Project_Backend/Model_File/DN21_model.h5")
# model.load_weights("D:/Notes semester-8/FYP-2_updated/Project_Backend/Model_File/updated_model_weights.h5") 


# def preprocess_image(image):
#     try:
#         logging.info("Preprocessing image...")
#         image = np.array(image)
#         image = image / 255.0
#         image = np.expand_dims(image, axis=0)
#         return image
#     except Exception as e:
#         logging.error(f"Error in preprocessing image: {str(e)}")
#         raise

# def predict(image):
#     try:
#         logging.info("Predicting image...")
#         preprocessed_image = preprocess_image(image)
#         predictions = model.predict(preprocessed_image)
#         print(predictions)
#         print(np.argmax(predictions[0]))
#         predicted_class_index = np.argmax(predictions[0])
#         predicted_class = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'][predicted_class_index]
#         confidence = float(predictions[0][predicted_class_index])
#         logging.info(f"Predictions: {predictions}")
#         logging.info(f"Predicted class: {predicted_class}, Confidence: {confidence}")
#         return predicted_class, confidence
#     except Exception as e:
#         logging.error(f"Error in predicting: {str(e)}")
#         raise

# @app.route('/predict', methods=['GET', 'POST'])
# def predict_tumor():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return jsonify({'error': 'No file part'})

#         file = request.files['file']
#         if file.filename == '':
#             return jsonify({'error': 'No selected file'})
        
#         logging.info(f"Received file: {file.filename}")

#         if file.filename.lower().endswith('.jpg'):
#             try:
#                 image = Image.open(file)
#                 predicted_class, confidence = predict(image)
#                 return jsonify({'predicted_class': predicted_class, 'confidence': confidence})
#             except Exception as e:
#                 logging.error(f"Error during prediction: {str(e)}")
#                 return jsonify({'error': str(e)})
#         else:
#             return jsonify({'error': 'Invalid file format. Must be jpg'})

#     elif request.method == 'GET':
#         return jsonify({'message': 'Send a POST request with an image file to predict brain tumor.'})

# if __name__ == '__main__':
#     app.run(host='localhost', port=5000, debug=True)





# from flask import Flask, request, jsonify
# import numpy as np
# import tensorflow as tf
# from PIL import Image
# from io import BytesIO

# # Initialize Flask application
# app = Flask(__name__)

# # Load the saved model
# model_path = 'D:/Notes semester-8/FYP-2_updated/Project_Backend/Model_File/DN21_model.h5'  # Update the path to your saved model
# loaded_model = tf.keras.models.load_model(model_path)
# classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# # Define a function to preprocess the image
# def preprocess_image(image):
#     img = image.resize((224, 224))  # Resize the image to match the model input size
#     img_array = np.array(img)
#     # img_array = img_array / 255.0  # Normalize pixel values
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# # Define a route for prediction
# @app.route('/predict', methods=['POST','GET'])
# def predict():

#     # Check if an image file is included in the request
#     if request.method == 'POST':
#         if 'image' not in request.files:
#             return jsonify({'error': 'No image provided'})

#     # Read the image file from the request
#         image = request.files['image'].read()
    
#     # Convert image bytes to PIL Image object
#         img = Image.open(BytesIO(image))
    
#     # Preprocess the image
#         img_array = preprocess_image(img)
    
#     # Make prediction
#         prediction = loaded_model.predict(img_array)
#         predicted_class = classes[np.argmax(prediction)]
    
#         return jsonify({'prediction': predicted_class}), 200

# # Run the Flask application
# if __name__ == '__main__':
#     app.run(host='localhost', port=5000, debug=True)





# from flask import Flask, request, jsonify
# import numpy as np
# import tensorflow as tf
# from PIL import Image
# from io import BytesIO

# # Initialize Flask application
# app = Flask(__name__)

# # Load the saved model
# model_path = 'D:/Notes semester-8/FYP-2_updated/Project_Backend/Model_File/DN21_model.h5'  # Update the path to your saved model
# loaded_model = tf.keras.models.load_model(model_path)
# classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# # Define a function to preprocess the image
# def preprocess_image(image):
#     img = image.resize((224, 224))  # Resize the image to match the model input size
#     img_array = np.array(img)
#     # img_array = img_array / 255.0  # Normalize pixel values
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array


# @app.route('/predict', methods=['POST', 'GET'])
# def predict():
#     if request.method == 'POST':
#         # Check if an image file is included in the request
#         if 'image' not in request.files:
#             return jsonify({'error': 'No image provided'}), 400

#         # Read the image file from the request
#         image = request.files['image'].read()
    
#         # Convert image bytes to PIL Image object
#         img = Image.open(BytesIO(image))
    
#         # Preprocess the image
#         img_array = preprocess_image(img)
    
#         # Make prediction
#         prediction = loaded_model.predict(img_array)
#         predicted_class = classes[np.argmax(prediction)]
    
#         return jsonify({'prediction': predicted_class}), 200
#     else:
#         return jsonify({'error': 'Invalid request method. Use POST method to send image'}), 405



# # Run the Flask application
# if __name__ == '__main__':
#     app.run(host='localhost', port=5000, debug=True)
