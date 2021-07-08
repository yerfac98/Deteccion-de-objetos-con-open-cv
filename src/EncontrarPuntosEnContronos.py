'''
Encontrar puntos extremos en contornos con OpenCV

'''
import imutils
import cv2

# cargue la imagen, conviértala a escala de grises y difumínela ligeramente
image = cv2.imread("manos.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# umbral de la imagen, luego realice una serie de erosiones +
# dilataciones para eliminar pequeñas regiones de ruido
thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)

# encuentre contornos en la imagen de umbral, luego tome el más grande
# uno
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)

# determinar los puntos más extremos a lo largo del contorno
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])

# dibuja el contorno del objeto, luego dibuja cada uno de los
# puntos extremos, donde el más a la izquierda es rojo, el más a la derecha
# es verde, la parte superior es azul y la parte inferior es verde azulado

cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
cv2.circle(image, extLeft, 8, (0, 0, 255), -1)
cv2.circle(image, extRight, 8, (0, 255, 0), -1)
cv2.circle(image, extTop, 8, (255, 0, 0), -1)
cv2.circle(image, extBot, 8, (255, 255, 0), -1)

# mostrar la imagen de salida
cv2.imshow("Image", image)
cv2.waitKey(0)
