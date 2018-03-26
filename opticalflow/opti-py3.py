import numpy as np
import argparse
import cv2
import time
from math import sqrt, pow, atan2


video_w=640
video_h=480

########### Mouse Click event ##################
def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global p0
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		mpt=np.array([[[x,y]]],dtype=np.float32)
		p0=np.concatenate((p0,mpt))
##################################################

########## Parameters Defination ###############
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 1000,
                       qualityLevel = 0.01,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Take first frame and find corners in it
cap = cv2.VideoCapture("soccer1.mp4")
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
#p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
flag=0
p0 = np.array([[[0.,0.]]], dtype=np.float32) ##initialize point
for i in range(0,video_w,20):
	for j in range(0,video_h,20):
		pt=np.array([[[i,j]]],dtype=np.float32)
		p0=np.concatenate((p0,pt))
cv2.namedWindow("frame")
#cv2.namedWindow("flow")
cv2.setMouseCallback("frame", click_and_crop)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter('output.avi', fourcc, 30.0, (video_w,video_h))
#out2 = cv2.VideoWriter('output_flow.avi', fourcc, 30.0, (1280,720))
frame_n = 0 
##################################################
ff=[0]*10
#################### Main Function ######################
while(cap.isOpened()):
	ret,frame = cap.read()
	ff.pop(0)
	ff.append(0)
	ff[9]=frame
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame_n = frame_n + 1
	print(frame_n)
	flow = np.zeros_like(old_frame)
	# calculate optical flow
	p1, st, err =cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
	## Select good points
	good_new = p1#[st==1]
	good_old = p0#[st==1]
	# draw the tracks
	if type(ff[0]) != int:
		f = open('output/frame_'+format(frame_n,'03d')+'.txt', 'w') 
		for i,(new,old) in enumerate(zip(good_new,good_old)):
			a,b = new.ravel()
			c,d = old.ravel()
			if flag == 0 :
		    		#flow = cv2.circle(flow,(a,b),3,[0,0,255],-1)
		    		flag == 1
			dis=sqrt(pow((a-c),2)+pow((b-d),2))
			degree = atan2(a-c,b-d)
			#if dis <=5 :
			if dis >0.5 and dis<10 :
				ff[0]=cv2.arrowedLine(ff[0],(c,d),(a,b),[0,0,255],1,8,0,0.5)
				f.writelines(str(i)+"\t"+str(degree)+"\t"+str(dis)+"\n")
		cv2.imshow('frame',ff[0])
		#cv2.imshow('flow',flow)
		k = cv2.waitKey(10) & 0xff
		if k == 27:
			break
		if k == ord("p"):
			time.sleep(2)
	    # Now update the previous frame and previous points
		old_gray = frame_gray.copy()
	    # Write file
		out.write(ff[0])
    #out.write(flow)

cv2.destroyAllWindows()
cap.release()
f.close()

