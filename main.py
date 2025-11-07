import cv2
from faceregcogcam import FaceRecognitionWebCam

def main():
    print("Face Recognition Application")
    print("Connect iPhone via USB (using iVCam or similar)")
    print("\nPress Enter to auto-detect or enter camera index (0-9): ", end='')
    user_input = input().strip()
    
    try:
        app = FaceRecognitionWebCam()
        
        if user_input.isdigit():
            camera_idx = int(user_input)
            app.video_capture.release()
            app.video_capture = cv2.VideoCapture(camera_idx)
            if not app.video_capture.isOpened():
                print(f"Could not open camera {camera_idx}")
                return
        
        app.run()
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()