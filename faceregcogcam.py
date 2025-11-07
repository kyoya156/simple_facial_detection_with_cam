import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

class FaceRecognitionWebCam:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True

        self.load_known_faces()

        cam_index = self.find_camera()
        self.video_capture = cv2.VideoCapture(cam_index)

        if not self.video_capture.isOpened():
            raise Exception("Could not open camera")

    def find_camera(self):
        for index in range(10):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cap.release()
                    print(f"Camera found at index {index}")
                    return index
            cap.release()
        print("No camera found, using index 0")
        return 0

    def load_known_faces(self):
        known_faces_dir = 'known_faces'

        if not os.path.exists(known_faces_dir):
            os.makedirs(known_faces_dir)
            print(f"No known faces found. Created directory '{known_faces_dir}'. Please add known face images.")
            print(f"Please add known face images.")
        
        if not os.listdir(known_faces_dir):  # ✅ Then check if empty
            print(f"Directory '{known_faces_dir}' is empty. All faces will be tagged unknown.")
            return
        
        for filename in os.listdir(known_faces_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(known_faces_dir, filename)
                try:
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        self.known_face_encodings.append(encodings[0])
                        name = os.path.splitext(filename)[0].replace('_', ' ')
                        self.known_face_names.append(name)
                        print(f"Loaded: {name}")
                
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")

        print(f"Loaded {len(self.known_face_names)} known faces")

    def detect_and_recognize_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        self.face_locations = face_recognition.face_locations(
            rgb_small_frame,
            model='hog',
            number_of_times_to_upsample=1)
        
        self.face_encodings = face_recognition.face_encodings(
            rgb_small_frame,
            self.face_locations,
            num_jitters=1)
        
        self.face_names = []

        for face_encoding in self.face_encodings:
            name = "Unknown"
            confidence = 0.0
            
            if len(self.known_face_encodings) > 0:
                matches = face_recognition.compare_faces(
                    self.known_face_encodings,
                    face_encoding,
                    tolerance=0.6)

                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, 
                    face_encoding)
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = 1 - face_distances[best_match_index]
                    
            self.face_names.append((name, confidence))


    def draw_face_boxes(self, frame):
        if len(self.face_locations) != len(self.face_names):
            print(f"Warning: Mismatch - {len(self.face_locations)} faces but {len(self.face_names)} names")
            # Fill missing names with "Unknown"
            while len(self.face_names) < len(self.face_locations):
                self.face_names.append(("Unknown", 0.0))

        for (top, right, bottom, left), name_data in zip(self.face_locations, self.face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            name = "Unknown"
            confidence = 0.0

            if isinstance(name_data, tuple):
                name, confidence = name_data
            else:
                name = name_data

            color = (0, 255, 0) if name_data != "Unknown" else (0, 0, 255)
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)

            if name != "Unknown" and confidence > 0:
                label = f"{name} ({confidence*100:.1f}%)"
            else:
                label = name
            
            # Draw a label with a name below the face
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
        return frame
    
    def draw_info_panel(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        recognized = sum(1 for name_data in self.face_names 
                        if (isinstance(name_data, tuple) and name_data[0] != "Unknown") 
                        or (isinstance(name_data, str) and name_data != "Unknown"))
        unknown = len(self.face_names) - recognized

        info_lines = [
            f"Total Faces Detected: {len(self.face_locations)}",
            f"Recognized: {recognized} | Unknown: {unknown}",
            f"Known Database: {len(self.known_face_names)} faces",
            "",
            "Q-Quit | S-Save | R-Reload Faces"
        ]
        
        y_offset = 35
        for line in info_lines:
            cv2.putText(
                frame, line, (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
            y_offset += 25
            
        return frame
    
    def save_frame(self, frame):
        if not os.path.exists('captured_images'):
            os.makedirs('captured_images')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"captured_images/face_recognition_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        return filename

    def cleanup(self):
        self.video_capture.release()
        cv2.destroyAllWindows()
        print("Program completed.")

    def run(self):
        print("\nFace Recognition Started")
        print("Controls: Q-Quit | S-Save | R-Reload Known Faces\n")
        try:
            while True:
                ret, frame = self.video_capture.read()

                if not ret:
                    print("Failed to grab frame")
                    break

                if self.process_this_frame:
                    self.detect_and_recognize_faces(frame)

                self.process_this_frame = not self.process_this_frame

                frame = self.draw_face_boxes(frame)
                frame = self.draw_info_panel(frame)

                cv2.imshow("Face Recognition", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                elif key == ord("s"):
                    self.save_frame(frame)
                elif key == ord("r"):
                    print("\nReloading known faces...")
                    self.known_face_encodings = []
                    self.known_face_names = []
                    self.load_known_faces()
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.cleanup()
