#!/usr/bin/env python
# coding: utf-8

# In[8]:


import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import Input, layers

print(tf.__version__)


# In[2]:


#Download and preprocess dataset:
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# Now separate out a validation set.
val_frac = 0.1
num_val_samples = int(len(train_images)*val_frac)
# choose num_val_samples indices up to the size of train_images, !replace => no repeats
val_idxs = np.random.choice(np.arange(len(train_images)), size=num_val_samples, replace=False)
trn_idxs = np.setdiff1d(np.arange(len(train_images)), val_idxs)
val_images = train_images[val_idxs, :,:,:]
train_images = train_images[trn_idxs, :,:,:]

val_labels = train_labels[val_idxs]
train_labels = train_labels[trn_idxs]

train_labels = train_labels.squeeze()
test_labels = test_labels.squeeze()
val_labels = val_labels.squeeze()

input_shape  = train_images.shape[1:]
train_images = train_images / 255.0
test_images  = test_images  / 255.0
val_images   = val_images   / 255.0
print("Training Images range from {:2.5f} to {:2.5f}".format(np.min(train_images), np.max(train_images)))
print("Test     Images range from {:2.5f} to {:2.5f}".format(np.min(test_images), np.max(test_images)))

print(train_images.shape)
print(len(train_labels))
print(test_images.shape)
print(len(test_labels))
print(val_images.shape)
print(len(val_labels))


# In[3]:


model1 = tf.keras.Sequential([
    Input(shape=input_shape),
    layers.Conv2D(32, kernel_size=(3,3), activation="relu", padding='same', strides=(2,2)),
    layers.Conv2D(64, kernel_size=(3,3), activation="relu", padding='same', strides=(2,2)),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)), 
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dense(10)
])
model1.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model1.summary()


# In[4]:


train_hist1 = model1.fit(train_images, train_labels, 
                         validation_data=(val_images, val_labels),
                         epochs=50)


# In[6]:


test_loss1, test_acc1 = model1.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc1)


# In[36]:


#Load, preprocess, and display test image
test_image_frog = cv2.imread("./test_image_frog.jpg")
test_image_frog = cv2.resize(test_image_frog, (32, 32))
test_image_frog = test_image_frog / 255.0
plt.figure()
plt.imshow(test_image_frog, cmap='Greys')
plt.colorbar()
plt.grid(False)

print("Input shape:", model1.input_shape)
test_image_frog = np.expand_dims(test_image_frog, axis=0)
print("Test image shape:", test_image_frog.shape)


# In[40]:


#Predict label
prediction = model1.predict_classes(test_image_frog)
print("Predicted label:", class_names[prediction[0]])


# In[56]:


model2 = tf.keras.Sequential([
    Input(shape=input_shape),
    layers.DepthwiseConv2D(kernel_size=(3,3), padding='same', strides=(2,2)),
    layers.Conv2D(32, kernel_size=(1,1), activation="relu", strides=(1,1)),
    layers.DepthwiseConv2D(kernel_size=(3,3), padding='same', strides=(2,2)),
    layers.Conv2D(64, kernel_size=(1,1), activation="relu", strides=(1,1)),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)), 
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dense(10)
])
model2.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model2.summary()


# In[57]:


train_hist2 = model2.fit(train_images, train_labels, 
                         validation_data=(val_images, val_labels),
                         epochs=50)


# In[58]:


test_loss2, test_acc2 = model2.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc2)


# In[59]:


#Predict label
prediction = model2.predict_classes(test_image_frog)
print("Predicted label:", class_names[prediction[0]])

