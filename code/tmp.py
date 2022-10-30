import cv2
video = "../video/front1.mp4"
cap = cv2.VideoCapture(video)

frame=0
index=0
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    cv2.imshow("VideoFrame",image)
    cap.set(cv2.CAP_PROP_POS_FRAMES, path[index])
    index++
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
