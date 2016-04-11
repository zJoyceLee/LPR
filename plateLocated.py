import cv2
import numpy as np

def sobel(image, ksizeN):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = ksizeN)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = ksizeN)
    abs_grad_x = cv2.convertScaleAbs(sobelx)
    abs_grad_y = cv2.convertScaleAbs(sobely)
    grad = cv2.addWeighted(abs_grad_x, 0.5,  abs_grad_y, 0.5, 0)
    cv2.imshow('grad', grad)

    return grad


img = cv2.imread('./tmp/1.png')
# img = cv2.imread('./tmp/2.jpg')

blur = cv2.GaussianBlur(img, (5, 5), 0)

blur_gray = cv2.cvtColor( blur, cv2.COLOR_RGB2GRAY)

# laplacian = cv2.Laplacian(blur_gray, cv2.CV_64F)

grad = sobel(blur_gray, 3)

ret, binaryImg = cv2.threshold(grad, 127, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)

kernel = np.ones((3, 17), np.uint8)
closing = cv2.morphologyEx(binaryImg, cv2.MORPH_CLOSE, kernel)
cv2.imshow('closing', closing)

# image, contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# image =  cv2.drawContours(image, contours, 3, (15, 15, 234))
# cv2.imshow('contours', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
