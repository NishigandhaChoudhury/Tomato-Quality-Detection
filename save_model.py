import tensorflow as tf

model = tf.keras.models.load_model('models/tomato_model.keras')
model.save('models/tomato_model_compat.h5', save_format='h5')
print("Done! Model saved.")