import cv2
import img as img
from args import args

img = cv2.imread("centro.png")
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# convert the grayscale image to binary image

ret, thresh = cv2.threshold(gray_image, 127, 255, 0)

# calculate moments of binary image

M = cv2.moments(thresh)

# calculate x,y coordinate of center

cX = int(M["m10"] / M["m00"])

cY = int(M["m01"] / M["m00"])

# put text and highlight the center

cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
cv2.putText(img, "centro", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# display the image

cv2.imshow("Image", img)

cv2.waitKey(0)
