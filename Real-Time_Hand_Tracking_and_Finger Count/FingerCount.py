# Import Essential Libraries
import cv2
import time
import os
import HandDetectorModule as hdm

wCam, hCam = 640, 480  # --> Width and Height of the camera.

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folder_path = r"D:\Projects\Real-Time_Hand_Tracking_and_Finger Count\finger_images"

img_ls = os.listdir(folder_path)
print(img_ls)
overlay_ls = []

for imPath in img_ls:
    image_path = os.path.join(folder_path, imPath)
    img = cv2.imread(image_path)
    overlay_ls.append(img)

print(len(overlay_ls))
# cv2.imshow("img", img)
# cv2.waitKey(0)

pTime = 0

detector = hdm.handDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]
# 4 = Thumb, 8 = IndexFinger, 12 = MidFinger, 16 = RingFinger, 20 = PinkyFinger
while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lm_ls = detector.findPosition(img, draw=False)

    # print(lm_ls)

    if len(lm_ls) != 0:
        fingers = []
        if lm_ls[tipIds[0]][1] > lm_ls[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 fingers
        for id in range(1, 5):
            if lm_ls[tipIds[id]][2] < lm_ls[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        total_fingers = fingers.count(1)
        print(total_fingers)
        if not success:
            break  # Break the loop if the frame was not captured successfully

        h, w, c = overlay_ls[total_fingers - 1].shape

        # Point Calculation for count rectangle
        frame_height, frame_width, _ = img.shape
        rect_height = 50
        y1 = frame_height - rect_height
        y2 = frame_height
        x1 = 0
        x2 = frame_width

        img[0:h, 0:w] = overlay_ls[total_fingers - 1]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), cv2.FILLED)

        cv2.putText(
            img,
            str(total_fingers),
            (30, frame_height - 15),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (255, 0, 0),
            3,
        )

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(
            img, f"FPS:{int(fps)}", (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 5
        )

    cv2.imshow("Img", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Wait for 1 ms and check if 'q' is pressed
        break

cap.release()  # Release the video capture
cv2.destroyAllWindows()  # Close all OpenCV windows
