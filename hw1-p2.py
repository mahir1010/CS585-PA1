import cv2
import numpy as np

img= cv2.imread('img.png')
grayscale = np.zeros(img.shape[:-1],dtype=np.uint8)
for i in range(grayscale.shape[0]):
    for j in range(grayscale.shape[1]):
        grayscale[i][j]=np.average(img[i][j])
padded = np.pad(grayscale,((1,1),(1,1)))
print(padded.shape)
sobel_H=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sobel_V=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
sobel_hi=np.zeros_like(grayscale)
sobel_vi=np.zeros_like(grayscale)
for i in range(1,img.shape[0]):
    for j in range(1,img.shape[1]):
        sobel_hi[i][j]=np.sum(np.matmul(padded[i-1:i+2,j-1:j+2],sobel_H))
        sobel_vi[i][j] = np.sum(np.matmul(padded[i-1:i+2,j-1:j+2], sobel_V))
out= (sobel_vi+sobel_hi)//2
out = (out>100) * 255
cv2.imwrite('grayscale.jpg',grayscale)
cv2.imwrite("sobelConv.jpg",out.astype(np.uint8))