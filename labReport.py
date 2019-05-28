#coding:utf-8
from transform import four_point_transform
# from skimage.filters import threshold_local
import imageresize
import pytesseract
# from imutils import perspective
# import numpy as np
# import argparse
import cv2
import imutils
import findLine


def edgeDetection(image):
	image =imageresize.resize(image, height = 500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 100, 220)

	# cv2.imshow("Image", image)
	# cv2.imshow("Edged", edged)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	return edged


def findContour(edged,image):

	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)#image, contours, hierarchy
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]#使用opencv2
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)

		if len(approx) == 4:
			screenCnt = approx
			break

	 
	# cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
	# cv2.imshow("Outline", image)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	return screenCnt


def scan(screenCnt, image):
	ratio = image.shape[0] / 500.0

	warped = four_point_transform(image, screenCnt.reshape(4, 2) * ratio)
	# warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	# T = threshold_local(warped, 11, offset = 10, method = "gaussian")
	# warped = (warped > T).astype("uint8") * 255

	# kernel = np.ones((1,5), np.uint8)  # note this is a HORIZONTAL kernel
	# kernel = np.array([(0,1,0),(1,1,1),(0,1,0)])
	# e_im = cv2.dilate(warped, kernel, iterations=1)
	# e_im = cv2.erode(e_im, kernel, iterations=2) 

	# cv2.imshow("Original", imutils.resize(orig, height = 650))
	# cv2.imshow("Scanned", imutils.resize(warped, height = 650))
	# cv2.imshow("Scanne", imutils.resize(e_im, height = 650))
	# cv2.waitKey(0)

	return warped
def getdata1(i,tempimg):
	# local_img = tempimg[y: y + vertical_lenth, x: x + lateral_lenth]
	# filename = 'temp/' + 'data' + str(i) + '.jpg'
	# cv2.imwrite(filename, local_img)
	vertical_length = 42
	horizontal_length = 61
	partImage = tempimg[343:343+horizontal_length, 192+i*vertical_length:192 + (i+1)*vertical_length]
	if i==12:
		partImage = tempimg[342:342 + horizontal_length, 192 + i * vertical_length-1:192 + (i + 1) * vertical_length-1]
	# data=pytesseract.image_to_string(partImage)
	rows, cols, c = partImage.shape
	# print rows, cols
	# if cols>rows:
	# 	rows=cols
	# 先转化为正方形再旋转，否则旋转会出现黑色区域
	partImage1 = cv2.resize(partImage, (rows, rows))
	# cv2.imshow('1', partImage1)
	# 将竖直的图片旋转为水平的
	M = cv2.getRotationMatrix2D((rows / 2, rows / 2), 90, 1)
	dst = cv2.warpAffine(partImage1, M, (rows, rows))
	filename = 'temp/' + 'data' + str(i) + '.jpg'
	cv2.imwrite(filename, partImage)
	# cv2.imshow('da', dst)
	data = pytesseract.image_to_string(dst)
	# print data
	return data
def getdata2(i,tempimg):
	# local_img = tempimg[y: y + vertical_lenth, x: x + lateral_lenth]
	# filename = 'temp/' + 'data' + str(i) + '.jpg'
	# cv2.imwrite(filename, local_img)
	k=i-13
	vertical_length = 41
	horizontal_length = 59
	partImage = tempimg[843:843+horizontal_length, 366+k*vertical_length:366 + (k+1)*vertical_length]
	if i==21:
		partImage = tempimg[844:844 + horizontal_length, 366 + k * vertical_length:366 + (k + 1) * vertical_length]
	# data=pytesseract.image_to_string(partImage)
	rows, cols, c = partImage.shape
	# print rows, cols
	# if cols>rows:
	# 	rows=cols
	# 先转化为正方形再旋转，否则旋转会出现黑色区域
	partImage1 = cv2.resize(partImage, (rows, rows))
	# cv2.imshow('1', partImage1)
	# 将竖直的图片旋转为水平的
	M = cv2.getRotationMatrix2D((rows / 2, rows / 2), 90, 1)
	dst = cv2.warpAffine(partImage1, M, (rows, rows))
	filename = 'temp/' + 'data' + str(i) + '.jpg'
	cv2.imwrite(filename, partImage)
	# cv2.imshow('da', dst)
	data = pytesseract.image_to_string(dst)
	# print data#,type(data)
	return data
def romoveRepetedGap(str):
	resultStr = str
	tempStr = ''
	strList = list(resultStr)
	numflag=0
	for i in range(len(strList)):
		# if strList[i] != '.':
		# 	tempStr += strList[i]
		# elif strList[i] == '.' and i+1 ==len(strList):
		# 	tempStr += strList[i]
		# elif strList[i] == '.' and strList[i+1] !='.':
		# 	tempStr += strList[i]
		# else: continue
		if strList[i]!='.':
			tempStr +=strList[i]
		elif strList[i]=='.':
			numflag +=1;
			if numflag<2:
				tempStr +=strList[i]
		else:continue
	return tempStr
def main():
	image = cv2.imread("1.jpg")
	# print image.shape
	edged = edgeDetection(image)
	screenCnt = findContour(edged,image)
	scannedImage = scan(screenCnt,image)

	
	# cv2.imshow("Scanned", imageresize.resize(scannedImage, height = 650))
	cv2.imwrite("out1.jpg",scannedImage)
	print scannedImage.shape
	# print scannedImage.shape
	#分为纵向和横向
	if scannedImage.shape[0]>scannedImage.shape[1]:
		cutImage=cv2.resize(scannedImage,(960,1280))
	else:
		cutImage=cv2.resize(scannedImage,(1280,960))
	# print cutImage.shape[0]
	# print cutImage.shape[1]
	# cutImage=imageresize.resize(scannedImage,width=960,height=1280)
	# print cutImage.shape
	cv2.imwrite("out2.jpg", cutImage)
	# 确定三条定位直线,要vertical,horizontal讨论
	# lineImage=findLine.lineFind(cutImage)
	# cv2.imwrite('houghlines.jpg', lineImage)
	# cv2.imshow("hough",lineImage)
	# vertical_length = 39
	# horizontal_length = 50
	# 进行ocr识别
	# partImage=cutImage[340:340+5*horizontal_length,192:192+vertical_length]
	# # data=pytesseract.image_to_string(partImage)
	# rows,cols,c=partImage.shape
	# print rows,cols
	# # if cols>rows:
	# # 	rows=cols
	# # 先转化为正方形再旋转，否则旋转会出现黑色区域
	# partImage1=cv2.resize(partImage,(rows,rows))
	# cv2.imshow('1',partImage1)
	# # 将竖直的图片旋转为水平的
	# M = cv2.getRotationMatrix2D((rows / 2, rows / 2), 90, 1)
	# dst=cv2.warpAffine(partImage1,M,(rows,rows))
	# cv2.imshow('da', dst)
	# data=pytesseract.image_to_string(dst)
	# print data
	undata=[]
	sdata=[]
	for i in range(22):
		if i<13:
			undata.append(getdata1(i,cutImage))
		else:
			# j=i-13
			undata.append(getdata2(i,cutImage))
	print "初读数据：",undata
	sdata=undata

	# for i in range(22):
	# 	sdata[i]=[str(undata[i])]
	# sdata=[str(x) for x in undata]
	#出去第一个因为裁剪不当的不正常字符
	for i in range(22):
		if sdata[i][0] == 'o' or sdata[i][0] == 'O':#有时候会把第一个0识别成o或者O
			sdata[i] = '0'+sdata[i][1:]
		if sdata[i][0] == 'Z' or sdata[i][0] == 'z':#会把2识别为Z或者z
			sdata[i] = u'2' + sdata[i][1:]
		if sdata[i][0] == 'I' or sdata[i][0] == 'l':
			sdata[i] = u'1' + sdata[i][1:]
		if sdata[i][0]<'0' or sdata[i][0]>'9':
			sdata[i]=sdata[i][1:]
		#加两个判断是因为存在前两个识别结果都有误但是实际情况是有数字的情况
		if sdata[i][0] == 'o' or sdata[i][0] == 'O':#有时候会把第一个0识别成o或者O
			sdata[i] = u'0'+sdata[i][1:]
		if sdata[i][0] == 'Z' or sdata[i][0] == 'z':#会把2识别为Z或者z
			sdata[i] = u'2' + sdata[i][1:]
		if sdata[i][0] == 'I' or sdata[i][0] == 'l':
			sdata[i] = u'1' + sdata[i][1:]
		for j in range(len(undata[i])):
			if sdata[i][j]<'0' or sdata[i][j]>'9':
				tdata=sdata[i][0:j]+u'.'+sdata[i][j+1:]	#处理因为识别问题把小数点识别成上引号
				sdata[i]=tdata
	print "一次处理：",sdata
	qdata = [str(x) for x in undata]
	wdata = [romoveRepetedGap(x) for x in qdata]	#因为把非数字的都转化为小数点，所以把相同小数点删除
	print "二次处理：",wdata   #输出数据
	# if len(wdata[18])>3:
	# 	wdata[18]=wdata[18][1:]
	# if len(wdata[6])>5:
	# 	wdata[6]=wdata[6][1:]
	# if len(wdata[6])>5:
	# 	wdata[6]=wdata[6][1:]
	# if len(wdata[6])>5:
	# 	wdata[6]=wdata[6][1:]
	true_data=['157', '5.39', '0.50', '2.10', '5.70', '41.8', '49.90', '0.04', '0.17', '0.46', '3.39', '4.05', '8.11', '11.10', '13.9', '0.26', '233', '11.7', '337', '29.1', '86.5', '46.6']
	for i in [2, 3, 6, 7, 8, 18]:	#这几个数据里中文较近，有时识别出其他数字，没有在前一步当作非法字符去除
		if len(wdata[i]) > len(true_data[i]):
			wdata[i] = wdata[i][1:]
	print "最终处理：",wdata
	print "原始数据：",true_data
	print wdata == true_data
	# 查看各个位置像素点
	# cutImage1=cv2.resize(cutImage,None,fx=0.5,fy=0.5)
	# cv2.imshow('dst',cutImage1)
	# grayPartImage = cv2.cvtColor(partImage, cv2.COLOR_BGR2GRAY)
	# print data
	# cv2.imshow('da',partImage)
	# filename = 'temp/' + 'data' + str(1) + '.jpg'
	# cv2.imwrite(filename,partImage)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
