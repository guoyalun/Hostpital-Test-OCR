#coding:utf-8
import cv2
import numpy as np
# import matplotlib.pyplot as plt
def lineFind(img):
	# img=cv2.imread()
	# gray=cv2.cvtColor(img,cv2.COLOR_BAYER_BG2BGR)
	# gaus=cv2.GaussianBlur(gray,(3,3),0)
	edges=cv2.Canny(img,150,200,apertureSize=3)
	lines=cv2.HoughLines(edges,1,np.pi/180,300)
	lines1=lines[:,0,:]
	for line in lines:
	# print(line)

		rho = line[0][0]
		theta = line[0][1]
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a * rho
		y0 = b * rho
		x1 = int(x0 + 10 * (-b))
		y1 = int(y0 + 10 * (a))
		x2 = int(x0 - 1200 * (-b))
		y2 = int(y0 - 1200 * (a))
		cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
		print x1, y1, x2, y2
	# cv2.imshow(("hough", img))
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	return img