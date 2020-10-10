import tensorflow as tf
from PIL import Image
import tensorflow_hub as hub
from utils import process_image, get_class_names, load_model

import argparse
import numpy as np
import json

def predict(image_path, model, top_k,class_names):
    image = Image.open(image_path)
    image_array = np.asarray(image)
    processed_image = process_image(image_array)
    
    
    prediction = model.predict(np.expand_dims(processed_image, 0))
    
    values, indices= tf.math.top_k(prediction[0].tolist(), k=top_k)
    probability = values.numpy().tolist()
    labels = [class_names[str(i)] for i in indices.numpy().tolist()]
    
    print("Number of Top values to show(K): ",top_k)
    print("Predicted Flower: ",labels[0])
    print("Predicted Probabilities:",probability)
    print("Predicted Classes:",indices.numpy().tolist())
    print('Predicted Labels:',labels)
    
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Image Classsifier parser")
    parser.add_argument("image_path",help="Path of the image to be predicted")
    parser.add_argument("saved_model",help="Path of the saved_model")
    parser.add_argument("--top_k", help="Show top k prediction", required = False, default = 5, type = int)
    parser.add_argument("--category_names", help="Json File path for saved labels ", required = False, default = "label_map.json")
    args = parser.parse_args()
    
    model = load_model(args.saved_model)
    class_names = get_class_names(args.category_names)
    predict(args.image_path, model, args.top_k, class_names)
