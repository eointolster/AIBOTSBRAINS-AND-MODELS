import tensorflow as tf
import numpy as np

def cosine_similarity_loss(y_true, y_pred):
    return -tf.reduce_sum(tf.nn.l2_normalize(y_true, axis=-1) * tf.nn.l2_normalize(y_pred, axis=-1), axis=-1)

class NormalizationLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.nn.l2_normalize(inputs, axis=-1)

model_path = 'direction_model_3d.h5'

# Load the model without compiling
model = tf.keras.models.load_model(model_path, compile=False)

# Recompile the model with the correct loss function
model.compile(optimizer='adam', loss='mse')  # Use 'mse' instead of custom loss

# Test prediction
test_input = np.random.rand(1, 6)  # Assuming input shape is (1, 6)
prediction = model.predict(test_input)
print("Test prediction shape:", prediction.shape)
print("Test prediction:", prediction)