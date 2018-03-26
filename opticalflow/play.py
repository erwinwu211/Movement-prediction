import numpy as np
import argparse
import cv2
import time
from math import sqrt, pow, atan2



cap = cv2.VideoCapture("output.avi")

while(cap.isOpened()):
    ret,frame = cap.read()
    
    cv2.imshow('frame',frame)

    cv2.waitKey(-1)

cv2.destroyAllWindows()
cap.release()

############################################################
