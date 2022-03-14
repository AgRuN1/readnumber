import cv2
import numpy as np
import imutils
import easyocr
from matplotlib import pyplot as plt


#read the image
img = cv2.imread('Images/i.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#filtered = cv2.bilateralFilter(gray, 11, 15, 15)


#get contours
edges = cv2.Canny(gray, 30, 200)
conts = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
conts = imutils.grab_contours(conts)
conts = sorted(conts, key=cv2.contourArea, reverse=True)[:8]


#find number's contour
pos = None
for cont in conts:
    approx = cv2.approxPolyDP(cont, 10, True)
    if len(approx) == 4:
        pos = approx
        break


#get number's image
mask = np.zeros(gray.shape, np.uint8)
new_img = cv2.drawContours(mask, [pos], 0, 255, -1)
bitwise_img = cv2.bitwise_and(img, img, mask=mask)
x, y = np.where(mask == 255)
x1, y1 = np.min(x), np.min(y)
x2, y2 = np.max(x), np.max(y)
cropped = gray[x1:x2, y1:y2]


#show number's image
plt.imshow(cropped)
plt.show()


#get the text
reader = easyocr.Reader(['en'])
result = reader.readtext(cropped)
text = result[0][-2][:6].upper().replace(' ', '')
print(text)