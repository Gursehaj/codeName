import cv2

background = cv2.resize(cv2.imread("../face.jpg"), (1280,720), interpolation= cv2.INTER_NEAREST)
top = cv2.resize(cv2.imread("../test.jpg", -1), (300,300), interpolation= cv2.INTER_NEAREST)
cv2.rectangle(background,(0,0),(510,510),(0,255,0),3)

x_offset=y_offset=50

y1, y2 = y_offset, y_offset + top.shape[0]
x1, x2 = x_offset, x_offset + top.shape[1]

alpha_s = top[:, :, 2] / 255.0
alpha_l = 1.0 - alpha_s

for c in range(0, 2):
    background[y1:y2, x1:x2, c] = (alpha_s * top[:, :, c] + alpha_l * top[y1:y2, x1:x2, c])

# x_offset=y_offset=50
# top
# background[y_offset:y_offset+top.shape[0], x_offset:x_offset+top.shape[1],:] = top
cv2.imshow("img",background)
cv2.waitKey(0)