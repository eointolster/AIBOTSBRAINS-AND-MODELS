from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

class NormalizationLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.nn.l2_normalize(inputs, axis=-1)

def load_model_safe(model_path):
    try:
        model = tf.keras.models.load_model(model_path, 
                                           custom_objects={'cosine_similarity_loss': cosine_similarity_loss,
                                                           'NormalizationLayer': NormalizationLayer})
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Replace the model loading part with:
model_path = 'direction_model_3d.h5'
model_3d = None

if os.path.exists(model_path):
    print(f"Model file found at {model_path}")
    try:
        print("Attempting to load model...")
        model_3d = tf.keras.models.load_model(model_path, compile=False)
        model_3d.compile(optimizer='adam', loss='mse')
        print("Model loaded and compiled successfully.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
else:
    print(f"Model file not found at {model_path}")

print(f"model_3d is None: {model_3d is None}")

@app.route('/')
def index():
    return render_template('test_index.html')

@app.route('/test_prediction', methods=['POST'])
def test_prediction():
    if model_3d is None:
        print("Model is not loaded.")
        return jsonify({'error': 'Model not loaded. Please train and save the model first.'})

    try:
        data = request.json['positions']
        start_pos = np.array(data[:3], dtype=np.float32)
        end_pos = np.array(data[3:], dtype=np.float32)

        # Normalize input for the model
        normalized_input = (np.array(data, dtype=np.float32) + 5) / 10  # Convert from [-5, 5] to [0, 1]
        normalized_input = normalized_input * 2 - 1  # Convert from [0, 1] to [-1, 1]

        # Make prediction
        prediction = model_3d.predict(np.array([normalized_input]))
        predicted_direction = prediction[0]

        # Calculate correct direction
        correct_direction = end_pos - start_pos
        correct_direction = correct_direction / np.linalg.norm(correct_direction)

        # Calculate cosine similarity
        similarity = cosine_similarity(predicted_direction, correct_direction)
        angle = np.arccos(np.clip(similarity, -1.0, 1.0))
        angle_degrees = np.degrees(angle)

        response_data = {
            'start_position': start_pos.tolist(),
            'end_position': end_pos.tolist(),
            'predicted_direction': predicted_direction.tolist(),
            'correct_direction': correct_direction.tolist(),
            'angle_difference': float(angle_degrees)
        }
        print("Test Prediction Response data:", response_data)
        return jsonify(response_data)
    except Exception as e:
        print(f"Error in test_prediction: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'})

@app.route('/random_test', methods=['GET'])
def random_test():
    if model_3d is None:
        print("Model is not loaded.")
        return jsonify({'error': 'Model not loaded. Please train and save the model first.'})

    try:
        # Generate random start and end positions in the range [-5, 5]
        start_pos = np.random.uniform(-5, 5, 3).astype(np.float32)
        end_pos = np.random.uniform(-5, 5, 3).astype(np.float32)

        # Normalize input for the model
        normalized_input = (np.concatenate([start_pos, end_pos]) + 5) / 10  # Convert from [-5, 5] to [0, 1]
        normalized_input = normalized_input * 2 - 1  # Convert from [0, 1] to [-1, 1]

        # Make prediction
        prediction = model_3d.predict(np.array([normalized_input]))
        predicted_direction = prediction[0]

        # Calculate correct direction
        correct_direction = end_pos - start_pos
        correct_direction = correct_direction / np.linalg.norm(correct_direction)

        # Calculate cosine similarity
        similarity = cosine_similarity(predicted_direction, correct_direction)
        angle = np.arccos(np.clip(similarity, -1.0, 1.0))
        angle_degrees = np.degrees(angle)

        response_data = {
            'start_position': start_pos.tolist(),
            'end_position': end_pos.tolist(),
            'predicted_direction': predicted_direction.tolist(),
            'correct_direction': correct_direction.tolist(),
            'angle_difference': float(angle_degrees)
        }
        print("Random Test Response data:", response_data)
        return jsonify(response_data)
    except Exception as e:
        print(f"Error in random_test: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True, port=5001)