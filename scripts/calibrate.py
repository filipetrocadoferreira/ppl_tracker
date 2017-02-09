import cv2
import numpy as np
import math
import os
import pickle

fileDir = os.path.dirname(os.path.realpath(__file__))
fileDir = os.path.join(fileDir,'..')
videoDir = os.path.join(fileDir,'Files')
mapFile  = os.path.join(videoDir,'map.png')
videoDir = os.path.join(videoDir,'Arc')
videoDir = os.path.join(videoDir,'5-actor')
videoDir = os.path.join(videoDir,'equal-colors')
cam0 = os.path.join(videoDir,'arc_mp_ec_C10.mp4')
cam1 = os.path.join(videoDir,'arc_mp_ec_C11.mp4')
cam2 = os.path.join(videoDir,'arc_mp_ec_C12.mp4')
cam3 = os.path.join(videoDir,'arc_mp_ec_C13.mp4')
cam = os.path.join(videoDir,'arc_mp_ec_C1')
calibfile = os.path.join(fileDir,'calibration')




img_points = [[]]

def callback_img(event,x,y,flags,param):
	global img_points
	if flags == cv2.EVENT_FLAG_LBUTTON:
		img_points[0].append((x,y))
		
		print img_points
		draw()

def callback_map(event,x,y,flags,param):
	global map_points
	if flags == cv2.EVENT_FLAG_LBUTTON:
		map_points[0].append((x,y))
		
		print map_points
		draw()

def process_polygon(img,points):
	maskImage=np.zeros_like(img)
	cv2.drawContours(maskImage,np.array(points),0,255,-1)
	cv2.imshow('extractedImage',maskImage)
	cv2.waitKey(-1)
	return maskImage



def draw():
	#draw points
	for i in (np.arange(1,len(img_points[0]))):
		cv2.line(img_show,img_points[0][i],img_points[0][i-1],(255,255,0),1)
		
	for i in (np.arange(1,len(map_points[0]))):
		cv2.line(map_show,map_points[0][i],map_points[0][i-1],(255,255,0),1)
	

##################################################### RUN SECTION ######################################

for i in (np.arange(0,4)):
	videocap = cv2.VideoCapture(cam+str(i)+'.mp4')
	ret,img_ori = videocap.read()
	img_show = img_ori.copy()
	img_ori = cv2.cvtColor(img_ori,cv2.COLOR_BGR2GRAY) 


	cv2.namedWindow('image')
	cv2.setMouseCallback('image',callback_img)

	while(1):
		cv2.imshow('image',img_show)
		k = cv2.waitKey(10) & 0xFF
		if k == ord('f'): #we're good with the polygon
			mask=process_polygon(img_ori,img_points)
			cv2.imwrite(os.path.join(calibfile,str(i)+'.bmp'),mask)
			img_points = [[]]
			break
			#~ 

##for cam0:

img_points = np.array([[769, 149], [364, 394], [558, 626], [1093, 261]]).astype(np.float32)
map_points = np.array([[0, 0], [6, 0], [6.5, 2.5], [0, 4.5]]).astype(np.float32)

print img_points
print map_points

M = cv2.getPerspectiveTransform(img_points,map_points)

print M
np.savetxt(os.path.join(calibfile,'0.txt'),M,delimiter=',')

##for cam1:

img_points = np.array([[137, 319], [343, 706], [975, 446], [440, 173]]).astype(np.float32)
map_points = np.array([[0, 0], [5.5, 0], [7, 4], [0, 4.5]]).astype(np.float32)

print img_points
print map_points

M = cv2.getPerspectiveTransform(img_points,map_points)

print M
np.savetxt(os.path.join(calibfile,'1.txt'),M,delimiter=',')

##for cam2:

img_points = np.array([[948, 642], [1106, 262], [784, 154], [173, 520]]).astype(np.float32)
map_points = np.array([[2.5, 0], [7.5, 0], [8, 4.25], [0, 4.5]]).astype(np.float32)

print img_points
print map_points

M = cv2.getPerspectiveTransform(img_points,map_points)

print M
np.savetxt(os.path.join(calibfile,'2.txt'),M,delimiter=',')

##for cam3:

img_points = np.array([[1124, 498], [495, 155], [174, 278], [390, 714]]).astype(np.float32)
map_points = np.array([[0, 0], [8, 0], [8, 4.25], [2, 4.25]]).astype(np.float32)

print img_points
print map_points

M = cv2.getPerspectiveTransform(img_points,map_points)

print M
np.savetxt(os.path.join(calibfile,'3.txt'),M,delimiter=',')



