import tensorflow as tf
import tensorflow_hub as hub
import json


IMAGE_SIZE = 224
IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

def get_class_names(json_file):
    with open(json_file, 'r') as f:
        class_names = json.load(f)
    # Class names contain index from 1 to 102, whereas the datasets have label indices from 0 to 101, hence     remapping
    updated_class_names = dict()
    for key in class_names:
        updated_class_names[str(int(key)-1)] = class_names[key]
    return updated_class_names


def load_model(model_path):
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
    print(model.summary())
    return model

def process_image(image):
    new_image = tf.image.convert_image_dtype(image, dtype=tf.int16, saturate=False)
    resized_image = (tf.image.resize(image,(IMAGE_SIZE,IMAGE_SIZE)).numpy())/255
    return resized_image