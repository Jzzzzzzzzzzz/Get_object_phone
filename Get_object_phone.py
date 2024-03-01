import cv2
import mediapipe as mp
import time
from imageai.Detection import ObjectDetection #pip install cython pillow>=7.0.0 numpy>=1.18.1 opencv-python>=4.1.2 torch>=1.9.0 --extra-index-url https://download.pytorch.org/whl/cpu torchvision>=0.10.0 --extra-index-url https://download.pytorch.org/whl/cpu pytest==7.1.3 tqdm==4.64.1 scipy>=1.7.3 matplotlib>=3.4.3 mock==4.0.3

capture = cv2.VideoCapture(0)
draw = mp.solutions.drawing_utils

mp_holistic = mp.solutions.holistic
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath("model/retinanet_resnet50_fpn_coco-eeacb38b.pth")
detector.loadModel()
output_path = "../cv2/image_web-video.jpg"

array_detection = []
count = 0
def phone_obj(obje,list=[]):
    global count
    print(count)
    if obje == "cell phone":
        start_phone = time.time()
        list.append(start_phone)
        print(start_phone)
        count += 1
        if count >= 4:
            finish_phone  = time.time()
            print(finish_phone-list[0])

            print(f"{int(finish_phone-list[0])}The person is not interested, he is on the phone")

finish = 0
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while capture.isOpened():
        ret, frame = capture.read()

        start = time.time()
        # print(start)
        if start - finish > 2:
            detections = detector.detectObjectsFromImage(input_image=frame, output_image_path=output_path,
                                                         minimum_percentage_probability=10)
            finish = time.time()

            for i in detections:
                print(i["name"])
                phone_obj(i["name"])

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # frame image
        # Make detections
        results = holistic.process(image)

        # Recolor image back to  BGE for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # frame image

        # Draw face landmarks
        draw.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                            draw.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                            draw.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

        # Right Hand
        draw.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                            draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                            draw.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

        # Left Hand
        draw.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                            draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                            draw.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))


        cv2.imshow("Webcam Feed", image)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

capture.release()
cv2.destroyAllWindows()

