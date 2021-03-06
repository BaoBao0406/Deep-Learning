from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys

# Load Completed Learning Module VGG16
model = VGG16(weights='imagenet')

# Function to identify image
def predict(filename, featuresize):
    img = image.load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(preprocess_input(x))
    results = decode_predictions(preds, top=featuresize)[0]
    return results

# Function to show image
def showing(filename, title, i):
    im = Image.open(filename)
    im_list = np.asarray(im)
    plt.subplot(2, 5, i)
    plt.title(title)
    plt.axis('off')
    plt.imshow(im_list)

# Identify Image1
filename = 'C:\\Users\\Patrick\\keras\\train\\cat.3591.jpg'
plt.figure(figsize=(20, 10))
for i in range(1):
    showing(filename, 'query', i+1)
plt.show()
results = predict(filename, 10)
for result in results:
    print(result)

# Identify Image2
filename = 'C:\\Users\\Patrick\\keras\\train\\dog.8035.jpg'
plt.figure(figsize=(20, 10))
for i in range(1):
    showing(filename, 'query', i+1)
plt.show()
results = predict(filename, 10)
for result in results:
    print(result)
    
# Identify Image3
filename = 'C:\\Users\\Patrick\\keras\\IMG_4706.jpg'
plt.figure(figsize=(20, 10))
for i in range(1):
    showing(filename, 'query', i+1)
plt.show()
results = predict(filename, 10)
for result in results:
    print(result)