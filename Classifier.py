import os 
import itertools 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, classification_report 

import tensorflow as tf 
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array 

# CONFIGURATIONS #
INPUT_DIR = "Data_Resized_Clean" # Directory containing processed images
NUM_CLASSES = 4 # Number of classes in the dataset
CLASS_NAMES = ["Prostate", "Skin", "Breast", "Control"]
BATCH_SIZE = 32 # Batch size for training
EPOCHS = 20 # Number of training epochs
IMG_SIZE = (224, 224) 
LR_P1 = 1e-4 # Learning rate for the optimizer
FINE_TUNE = 1e-5 

# IMAGE GENERATOR # 
train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
                                   rotation_range=5,
                                   vertical_flip=True,
                                   horizontal_flip=True)


val_test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)

train_generator = train_datagen.flow_from_directory(directory=os.path.join(INPUT_DIR, "Train"),
                                                    target_size=IMG_SIZE, 
                                                    batch_size=BATCH_SIZE, 
                                                    class_mode='categorical', 
                                                    shuffle=True) # Only shuffle train for SGD later, while we want val & test to be unshuffle for consistent validation 

val_generator = val_test_datagen.flow_from_directory(directory=os.path.join(INPUT_DIR, "Validation"),
                                                    target_size=IMG_SIZE, 
                                                    batch_size=BATCH_SIZE, 
                                                    class_mode='categorical', 
                                                    shuffle=False)
                        
test_generator = val_test_datagen.flow_from_directory(directory=os.path.join(INPUT_DIR, "Test"),
                                                    target_size=IMG_SIZE, 
                                                    batch_size=BATCH_SIZE, 
                                                    class_mode='categorical', 
                                                    shuffle=False) 

# CHECK THE DIMENSION #
print(f"Classes: {train_generator.class_indices}")
print(f"Train: {train_generator.samples} images")
print(f"Test: {test_generator.samples} images")
print(f"Val: {val_generator.samples} images")

# Check 1 batch to confirm shapes 
(x_batch, y_batch) = next(train_generator) 
print(f"\nBatch image shape: {x_batch.shape}") # (32, 224, 244, 3)
print(f"Batch label shape: {y_batch.shape}") # (32, 4)


# Build Model (Functional API) #
def build_model(inputs):
    base_model = ResNet50(input_shape=(224, 224, 3), 
                          include_top=False, 
                          weights="imagenet")
    
    # PHASE 1: Freeze the base first 
    base_model.trainable = False 
    x = base_model(inputs, training=False)  # Keeps BN layers frozen

    return x, base_model # return base_model so we can unfreeze later

def classifier_layers(inputs):
    x = layers.GlobalAveragePooling2D()(inputs)
    x = layers.Dropout(0.5)(x)                                  # Slightly increase dropout rate  to reduce the val/train gap
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return x

# Connect the model #
inputs = tf.keras.Input(shape=(224,224,3))
resnet_out, base_model = build_model(inputs)
outputs = classifier_layers(resnet_out)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.summary() 

# COMPILE & CALLBACKS - PHASE 1 #
model.compile(optimizer=optimizers.Adam(learning_rate=LR_P1, clipnorm=1.0), 
              loss="categorical_crossentropy",
              metrics=["accuracy"])

cb = [ callbacks.EarlyStopping(monitor="val_loss",patience=7, restore_best_weights=True),   
       callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)] # factor was 0.5, drop lr harder, after detected that learning was stagnate for val_loss (patience=5 specified # of epochs reduce after stagnate)

history1 = model.fit(train_generator, epochs=10, 
                     validation_data=val_generator,
                     callbacks=cb)

# UNFREEZE ^ FINETUNED WITH TINY LR - PHASE 2 #
base_model.trainable = True 

model.compile(optimizer=optimizers.Adam(learning_rate=FINE_TUNE, clipnorm=1.0), 
              loss="categorical_crossentropy", 
              metrics=["accuracy"])

# Update the learning rate without recompiling the model 
# model.optimizer.learning_rate.assign(FINE_TUNE)

history2 = model.fit(train_generator, epochs=15, 
                     validation_data=val_generator,
                     callbacks=cb)

base_model.trainable = True 
for layer in base_model.layers[:-30]:
    layer.trainable = False  

# ── Combine histories for plotting ───────────────────────────
def combine_histories(h1, h2, key):
    return h1.history[key] + h2.history[key]

for metric, val_metric, title, fname in [
    ("loss", "val_loss", "Loss per Epoch", "Loss_Plot.png"),
    ("accuracy", "val_accuracy", "Accuracy per Epoch", "Accuracy_Plot.png")
]:
    plt.plot(combine_histories(history1, history2, metric),     label=f"Train")
    plt.plot(combine_histories(history1, history2, val_metric), label=f"Val")
    plt.axvline(x=5, color='gray', linestyle='--', label='Fine-tune start')
    plt.xlabel("Epoch"); plt.ylabel(metric.capitalize())
    plt.title(title); plt.legend()
    plt.savefig(fname); plt.close()

# Saved the Model 
model.save("resnet50_cancer_classifier.keras") # Keras is the mordern TF format 

