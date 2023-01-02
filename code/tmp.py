import cv2
import numpy as np

image = np.zeros((512, 1024, 3), np.uint8) + 255
y_center = image.shape[0]//2
x_center = image.shape[1]//2

cv2.namedWindow('image')
cv2.circle(image, (x_center, y_center), 3, (255, 0, 0), -1)

infinity = 10000
image_height = image.shape[0]
image_width = image.shape[1]

for i in range(0, 100):

    angle = 0 + 45 * i
    if(angle >= 360):
        break
    # print(angle)

    # y, x 좌표를 각도 기준으로 회전
    # 기존 좌표와 회전한 좌표를 선으로 이었을 때, 이미지 끝 점과 교차점이 생기도록
    # 회전한 좌표는 이미지 밖에 두도록 합니다. 따라서 Length는 아주 큰 값 infinity로 둡니다.
    y_rotated = y_center + int(np.sin(np.pi / 180 * angle)*infinity)
    x_rotated = x_center + int(np.cos(np.pi / 180 * angle)*infinity)
    # print(y_rotated, x_rotated)
    
    # 기존 좌표와 이미지 끝에 교차된 점 까지를 이은 직선을 그립니다.
    ret, inner_point, clipped_point = cv2.clipLine((0, 0, image_width-1, image_height-1), (x_center, y_center), (x_rotated, y_rotated))
    cv2.line(image, (x_center, y_center), clipped_point, (255, 0, 0))

while(1):
    cv2.imshow('image',image)
    k = cv2.waitKey(1) & 0xFF
    
    if k == 27:
        break

cv2.destroyAllWindows()