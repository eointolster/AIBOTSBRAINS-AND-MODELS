import flask
from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import math
import threading
import time
import os

app = Flask(__name__)

# Custom loss function
def cosine_similarity_loss(y_true, y_pred):
    return -tf.reduce_sum(tf.nn.l2_normalize(y_true, axis=-1) * tf.nn.l2_normalize(y_pred, axis=-1), axis=-1)

# Custom normalization layer
class NormalizationLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.nn.l2_normalize(inputs, axis=-1)

# 2D Neural Network setup
model_2d = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model_2d.compile(optimizer='adam', loss='mse')

# 3D Neural Network setup
def create_model_3d():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(6,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(3),
        NormalizationLayer()
    ])

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=10000,
        decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(optimizer=optimizer, loss=cosine_similarity_loss)
    return model

model_3d = create_model_3d()

# Global variables
is_training_2d = False
is_training_3d = False
training_thread_2d = None
training_thread_3d = None
current_loss_2d = 0
current_loss_3d = 0
training_iterations_2d = 0
training_iterations_3d = 0

def generate_data_2d(num_samples):
    data = np.random.rand(num_samples, 4) * 28
    angles = np.arctan2(data[:, 3] - data[:, 1], data[:, 2] - data[:, 0])
    return data, angles

def generate_data_3d(num_samples):
    data = np.random.rand(num_samples, 6) * 2 - 1  # Range: -1 to 1
    directions = data[:, 3:] - data[:, :3]
    directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]  # Normalize
    return data, directions

def train_model_2d():
    global is_training_2d, current_loss_2d, training_iterations_2d
    while is_training_2d:
        X, y = generate_data_2d(100)
        history = model_2d.fit(X, y, epochs=1, verbose=0)
        current_loss_2d = history.history['loss'][0]
        training_iterations_2d += 1
        time.sleep(0.1)

def train_model_3d():
    global is_training_3d, current_loss_3d, training_iterations_3d
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    
    while is_training_3d:
        X, y = generate_data_3d(10000)
        history = model_3d.fit(X, y, epochs=1, batch_size=128, verbose=0, callbacks=[early_stopping])
        current_loss_3d = history.history['loss'][0]
        training_iterations_3d += 1
        time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

# 2D Model Routes
@app.route('/start_training', methods=['POST'])
def start_training():
    global is_training_2d, training_thread_2d, current_loss_2d, training_iterations_2d
    if not is_training_2d:
        is_training_2d = True
        current_loss_2d = 0
        training_iterations_2d = 0
        training_thread_2d = threading.Thread(target=train_model_2d)
        training_thread_2d.start()
    return jsonify({'message': '2D Training started'})

@app.route('/stop_training', methods=['POST'])
def stop_training():
    global is_training_2d
    is_training_2d = False
    if training_thread_2d:
        training_thread_2d.join()
    return jsonify({'message': '2D Training stopped'})

@app.route('/get_training_status', methods=['GET'])
def get_training_status():
    global current_loss_2d, training_iterations_2d
    return jsonify({
        'loss': current_loss_2d,
        'iterations': training_iterations_2d,
        'is_training': is_training_2d
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['positions']
    prediction = model_2d.predict(np.array([data]))
    correct_angle = math.atan2(data[3] - data[1], data[2] - data[0])
    is_correct = abs(prediction[0][0] - correct_angle) < 0.1
    return jsonify({
        'predicted_angle': float(prediction[0][0]),
        'correct_angle': float(correct_angle),
        'is_correct': bool(is_correct)
    })

@app.route('/save_model', methods=['POST'])
def save_model():
    model_2d.save('direction_model_2d.h5')
    return jsonify({'message': '2D Model saved'})

# 3D Model Routes
@app.route('/start_3d_training', methods=['POST'])
def start_3d_training():
    global is_training_3d, training_thread_3d, current_loss_3d, training_iterations_3d
    if not is_training_3d:
        is_training_3d = True
        current_loss_3d = 0
        training_iterations_3d = 0
        training_thread_3d = threading.Thread(target=train_model_3d)
        training_thread_3d.start()
    return jsonify({'message': '3D Training started'})

@app.route('/stop_3d_training', methods=['POST'])
def stop_3d_training():
    global is_training_3d
    is_training_3d = False
    if training_thread_3d:
        training_thread_3d.join()
    return jsonify({'message': '3D Training stopped'})

@app.route('/get_3d_training_status', methods=['GET'])
def get_3d_training_status():
    global current_loss_3d, training_iterations_3d
    return jsonify({
        'loss': current_loss_3d,
        'iterations': training_iterations_3d,
        'is_training': is_training_3d
    })

@app.route('/predict_3d', methods=['POST'])
def predict_3d():
    data = request.json['positions']
    prediction = model_3d.predict(np.array([data]))
    correct_direction = (np.array(data[3:]) - np.array(data[:3]))
    correct_direction /= np.linalg.norm(correct_direction)
    
    # Calculate angle between predicted and correct direction
    angle = np.arccos(np.clip(np.dot(prediction[0], correct_direction), -1.0, 1.0))
    angle_degrees = np.degrees(angle)
    
    is_correct = angle_degrees < 5  # Consider correct if angle is less than 5 degrees
    
    return jsonify({
        'predicted_direction': prediction[0].tolist(),
        'correct_direction': correct_direction.tolist(),
        'is_correct': bool(is_correct),
        'angle_difference': float(angle_degrees)
    })

@app.route('/save_3d_model', methods=['POST'])
def save_3d_model():
    model_3d.save('direction_model_3d.h5', save_format='h5')
    return jsonify({'message': '3D Model saved'})

@app.route('/load_3d_model', methods=['POST'])
def load_3d_model():
    global model_3d
    if os.path.exists('direction_model_3d.h5'):
        model_3d = tf.keras.models.load_model('direction_model_3d.h5', 
                                              custom_objects={'cosine_similarity_loss': cosine_similarity_loss,
                                                              'NormalizationLayer': NormalizationLayer})
        return jsonify({'message': '3D Model loaded successfully'})
    else:
        return jsonify({'message': 'No saved model found'})

if __name__ == '__main__':
    app.run(debug=True,port=5002)