import numpy as np
import os, json, re, time
import blickfeld_qb2
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0" # Makes connection way faster
import cv2 as cv

# Finding Data Save path from config file
f = open("config.json")
savePath = json.load(f)
savePath = savePath["dataPath"]
usIn = input("Save path is {}\nIf it looks right press enter, else type in correct path or fix config file and try again\n>>".format(savePath))
if len(usIn) != 0:
    savePath = usIn
# appending date to savepath
savePath = os.path.join(savePath, time.strftime("data-%m-%d-%Y",time.gmtime())) 
print("Using savepath: {}".format(savePath))
# making path if not exists
if not os.path.isdir(savePath):
    print("Path does not exist, creating",end='')
    os.makedirs(savePath,exist_ok=True)
    print(os.path.isdir(savePath))
# Checking if there are files in directory
preExistingFiles = os.listdir(savePath)
# Changing save index to one not in use
saveIndex = 1
if len(preExistingFiles) != 0:
    print("Files already in save directory.  ",end='')
    res = [int(re.search('_(\\d+)\\.',i).group(1)) for i in preExistingFiles if re.search('_(\\d+)\\.',i) is not None]
    saveIndex = np.max(res) + 1
print("saving at _{}".format(saveIndex))
        
time.sleep(10)


# Open camera, 0 is default, 1 may be the usb camera
cap = cv.VideoCapture(1)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 2160)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
ret, testFrame = cap.read()
if not ret:
    print("Camera failed to read")
print("Webcam dimensions: {}".format(testFrame.shape))

# Open LIDAR
with blickfeld_qb2.Channel(fqdn_or_ip="192.168.0.253") as channel:
    service = blickfeld_qb2.core_processing.services.PointCloud(channel)
    
    running = True
    num = 1
    while running:
        # Get frames
        lidarFrame = service.get().frame
        ret, camFrame = cap.read()
        
        # Print the frame ID
        print("Received frame with ID:", lidarFrame.id) 


        # Savedata
        np.save(savePath+"lidar_{}.npy".format(num),lidarFrame)
        cv.imwrite(savePath+"cam_{}.png".format(num),camFrame)
        print("Frames have been saved using number {}".format(num))
        usIn = input("Press enter to take another frame, type anything to quit0")
        if len(usIn) != 0:
            running = False


