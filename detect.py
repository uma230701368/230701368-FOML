import cv2
import numpy as np
import webbrowser
from tensorflow.keras.models import load_model

IMG_SIZE = (96, 96)

# Load trained model
model = load_model("emotion_model.h5")

# Class labels (must match folder names)
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Emotion → Tamil Song Mapping
emotion_music = {
    'happy': 'https://www.youtube.com/results?search_query=happy+tamil+songs',
    'neutral': 'https://www.youtube.com/results?search_query=calm+tamil+songs',
    'surprise': 'https://www.youtube.com/results?search_query=energetic+tamil+songs',
    'angry': 'https://www.youtube.com/results?search_query=intense+tamil+songs',
    'disgust': 'https://www.youtube.com/results?search_query=melancholic+tamil+songs',
    'fear': 'https://www.youtube.com/results?search_query=emotional+tamil+songs',
    'sad': 'https://www.youtube.com/results?search_query=sad+tamil+songs'
}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

print("Press 's' to open music recommendation")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, IMG_SIZE)
        face_img = face_img.astype("float32") / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        prediction = model.predict(face_img)
        emotion = classes[np.argmax(prediction)]

        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,0), 2)
        cv2.rectangle(frame, (x,y), (x+w,y+h),
                      (0,255,0), 2)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
            webbrowser.open(emotion_music[emotion])

    cv2.imshow("Emotion Music Recommendation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
