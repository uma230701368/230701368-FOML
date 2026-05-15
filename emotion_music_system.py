import os
import numpy as np
import cv2
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import webbrowser

#Dataset Preparation

# Image size for MobileNetV2
IMG_SIZE = (96, 96)
BATCH_SIZE = 32

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    'test',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Compute class weights to handle imbalance
classes = list(train_generator.class_indices.keys())
y_integers = train_generator.classes
class_weights_values = compute_class_weight(class_weight='balanced',
                                            classes=np.unique(y_integers),
                                            y=y_integers)
class_weights = dict(enumerate(class_weights_values))

#Build MobileNetV2 Model

# Base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96,96,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(7, activation='softmax')(x)  # 7 emotions

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base layers first
for layer in base_model.layers:
    layer.trainable = False

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


#Train the Model

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=25,  # you can increase later
    class_weight=class_weights
)

# Unfreeze some layers for fine-tuning (optional for better accuracy)
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_fine = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=15,
    class_weight=class_weights
)

#Evaluation metrics

# Predict on test set
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=classes))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))

#Real-Time Emotion Detection + Music Recommendation

# Map emotion → YouTube Tamil songs
emotion_music = {
    'happy': 'https://www.youtube.com/results?search_query=happy+tamil+songs',
    'neutral': 'https://www.youtube.com/results?search_query=calm+tamil+songs',
    'surprise': 'https://www.youtube.com/results?search_query=energetic+tamil+songs',
    'angry': 'https://www.youtube.com/results?search_query=intense+tamil+songs',
    'disgust': 'https://www.youtube.com/results?search_query=melancholic+tamil+songs',
    'fear': 'https://www.youtube.com/results?search_query=suspense+tamil+songs',
    'sad': 'https://www.youtube.com/results?search_query=sad+tamil+songs'
}

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, IMG_SIZE)
        face_img = face_img.astype('float32') / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        pred = model.predict(face_img)
        emotion = classes[np.argmax(pred)]

        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        # Open YouTube music link if 's' key pressed
        if cv2.waitKey(1) & 0xFF == ord('s'):
            webbrowser.open(emotion_music[emotion])

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

