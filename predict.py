import cv2
import numpy as np
from keras.models import load_model
from mtcnn.mtcnn import MTCNN

emotion_model = load_model("model_20241009-192827.h5")

emotion_labels = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

detector = MTCNN()


def detect_and_predict_emotions(frame):
    faces = detector.detect_faces(frame)

    for face in faces:
        x, y, width, height = face["box"]
        x2, y2 = x + width, y + height

        face_region = frame[y:y2, x:x2]

        face_region_resized = cv2.resize(face_region, (160, 160))
        face_region_resized = face_region_resized / 255.0  # Normalize pixel values

        face_input = np.expand_dims(face_region_resized, axis=0)

        predictions = emotion_model.predict(face_input)
        emotion_index = np.argmax(predictions)
        emotion_label = emotion_labels[emotion_index]

        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            emotion_label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (36, 255, 12),
            2,
        )

    return frame


def capture_and_process_video():
    cap = cv2.VideoCapture(0)
    print(f"camera opened: {cap.isOpened()}")
    while True:
        ret, frame = cap.read()

        if ret:
            frame_with_detections = detect_and_predict_emotions(frame)
            cv2.imshow("Emotion Detection", frame_with_detections)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


capture_and_process_video()
