import cv2
import face_recognition
import os
import datetime

# STEP 1: Load known faces
known_faces_dir = "known_faces"
known_encodings = []
known_names = []

# Create the folder if it doesn't exist
if not os.path.exists(known_faces_dir):
    os.makedirs(known_faces_dir)
    print("Place some images of known people inside the 'known_faces' folder before running.")
    print("Each image file name will be used as the person's name.")
    exit()

print("Loading known faces...")
for filename in os.listdir(known_faces_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(filename)[0])
        else:
            print(f"No face found in {filename}, skipped.")

if not known_encodings:
    print("No valid faces found in the 'known_faces' folder. Add at least one face image.")
    exit()

print(f"Loaded {len(known_encodings)} known face(s).")

# STEP 2: Setup output folder
output_dir = "recognized_faces"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# STEP 3: Start the webcam
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Cannot access webcam.")
    exit()

print("Webcam started. Press 'q' to quit.")

saved_faces = set()  # to prevent duplicate screenshots

# STEP 4: Real-time recognition loop
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Frame not captured. Exiting.")
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces and encodings in the frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare with known faces
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        # Find the closest match
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = face_distances.argmin() if len(face_distances) > 0 else None
        if best_match_index is not None and matches[best_match_index]:
            name = known_names[best_match_index]

        # Scale back up for original frame size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw box and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    
        # STEP 5: Save screenshot
    
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if name not in saved_faces:
            file_path = os.path.join(output_dir, f"{name}_{timestamp}.jpg")
            cv2.imwrite(file_path, frame)
            saved_faces.add(name)
            print(f"Saved: {file_path}")

    # Show the video window
    cv2.imshow("Face Recognition (Press 'q' to quit)", frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# STEP 6: Cleanup
video_capture.release()
cv2.destroyAllWindows()
print("Exited")
