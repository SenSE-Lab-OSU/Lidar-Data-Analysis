import numpy as np
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0" # Makes connection 
import cv2 as cv
import blickfeld_qb2
import time


# 
setFPS = 10
frameTime = 1/setFPS
running = True
savePath = "data/"
num = input("SelectNumber")


lidarFrameList = []
camFrameList = [] 



# Select camera, 0 is default, 1 may be the usb camera
cap = cv.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
ret, testFrame = cap.read()
if not ret:
    print("Camera failed to read")
print("Webcam dimensions: {}".format(testFrame.shape))


# I just copied the IP, is this the right thing?
with blickfeld_qb2.Channel(fqdn_or_ip="192.168.0.253") as channel:
    service = blickfeld_qb2.core_processing.services.PointCloud(channel)

    while running:
        old_time = time.monotonic() 
        print("frame at {}s".format(old_time))
        
        #Get frames -- commented out so it doesn't break without the actual stuff
        lidarFrame = service.get().frame
        ret, camFrame = cap.read()
        lidarFrameList.append(lidarFrame)
        camFrameList.append(camFrame)
        
        # Print the frame ID
        print("Received frame with ID:", lidarFrame.id)
    
        cur_time = time.monotonic()
        #time_to_pause = frameTime - (cur_time-old_time)
        #time.sleep(time_to_pause)
        running = False 


# Savedata
os.makedirs(savePath,exist_ok=True)
lidarArr = np.array(lidarFrameList)
camArr = np.array(camFrameList)
print("Frames have been saved")
np.save(savePath+"lidar_{}.npy".format(num),lidarArr)
print(camArr.shape)
cv.imwrite(savePath+"cam_{}.png".format(num),camArr[0,:,:,:])
np.save(savePath+"cam_{}.npy".format(num),camArr)

# TODO: MATLAB want their point clouds as a PCD file -> USE Open3d!

