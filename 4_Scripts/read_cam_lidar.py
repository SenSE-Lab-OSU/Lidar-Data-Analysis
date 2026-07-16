import numpy as np
from pypcd4 import PointCloud
import os, re, sys, blickfeld_qb2 

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0" # Significant speed improvements when this line is **BEFORE** import cv2 
import cv2 as cv

LIDAR_IP = sys.argv[2] if len(sys.argv) > 2 else "192.168.0.253"
SAVE_DIR = sys.argv[1] if len(sys.argv) > 1 else "./"
LIDAR_DIR = "lidar"
CAM_DIR = "camera"




def get_lastest_fnum():
    lidar_f = list(os.walk(os.path.join(SAVE_DIR,LIDAR_DIR)))[0][2] # Files only in top 
    cam_f = list(os.walk(os.path.join(SAVE_DIR,CAM_DIR)))[0][2]
    
    if len(lidar_f) == 0 and len(cam_f) == 0:
        # If the folders are both empty 
        return 0
    
    lpattern = r"lidar_([0-9]+)"
    lidar_numbers = []
    for file in lidar_f:
        fname = os.path.splitext(file)
        if fname[1] == ".npy":
            num = re.findall(lpattern, fname[0])
            if len(num) == 1:
                # file name is valid
                lidar_numbers.append(int(num[0]))
    cpattern = r"cam_([0-9]+)"
    cam_numbers = []
    for file in cam_f:
        fname = os.path.splitext(file)
        if fname[1] == ".png":
            num = re.findall(cpattern, fname[0])
            if len(num) == 1:
                # file name is valid
                cam_numbers.append(int(num[0]))
    lidar_numbers = np.sort(lidar_numbers)
    cam_numbers = np.sort(cam_numbers)
    if len(lidar_numbers) != len(cam_numbers):
        print("ERROR: Lidar and camera folders have an uneven number of saved files!")
        print("\t{}/{}: {} files".format(SAVE_DIR,LIDAR_DIR,len(lidar_numbers)))
        print("\t{}/{}: {} files".format(SAVE_DIR,CAM_DIR,len(cam_numbers)))
    else:
        for i in range(len(lidar_numbers)):
            if lidar_numbers[i] != cam_numbers[i]:
                print("ERROR: Missing either lidar or camera save")
                print("\tbad pair {}/lidar_{} and {}/cam_{}".format(LIDAR_DIR,lidar_numbers[i],CAM_DIR,cam_numbers[i]))
    return np.max(cam_numbers)


def take_data(lpath,cpath,save_id):
    # Open camera, 0 is default, 1 may be the usb camera
    print("Opening Camera ")
    cap = cv.VideoCapture(1)
    print("Camera Opened, Getting width and height: ")
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 3840)
    #cap.set(cv.CAP_PROP_FRAME_HEIGHT, 2160)
    cam_w = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    cam_h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    print("width={}, height={}".format(cam_w,cam_h))
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    ret, testFrame = cap.read()
    if not ret:
        print("Camera failed to read")
    print("Webcam dimensions: {}".format(testFrame.shape))

    # Open LIDAR
    with blickfeld_qb2.Channel(fqdn_or_ip=LIDAR_IP) as channel:
        service = blickfeld_qb2.core_processing.services.PointCloud(channel)
        
        id = save_id
        running = True
        while running:
            # Get frames
            lidarFrame = service.get().frame
            pcd = PointCloud.from_xyz_points(lidarFrame.binary.cartesian)
            ret, camFrame = cap.read()
            
            # Print the frame ID
            if not ret or not lidarFrame.id:
                print("Failed to recieve read lidar and camera:") 
            else:
                # Savedata
                lidar_save = os.path.join(lpath,"lidar_{}.npy".format(id))
                pcd_save = os.path.join(lpath,"lidar_{}.pcd".format(id))
                cam_save = os.path.join(cpath,"cam_{}.png".format(id))

                np.save(lidar_save,lidarFrame)
                pcd.save(pcd_save)
                cv.imwrite(cam_save,camFrame)
                
                print("Frames have been saved with id: {}".format(id))
                usIn = input("Press 't' to terminate, otherwise press enter to continue: ")
                if usIn == 't':
                    running = False
            id += 1



def main():
    full_lidar_dir = os.path.join(SAVE_DIR,LIDAR_DIR)
    full_cam_dir = os.path.join(SAVE_DIR,CAM_DIR)

    if not os.path.isdir(SAVE_DIR):
        print("{} does not exist, creating...".format(SAVE_DIR))
    # Creating directories either way, but just wanted to make it say something incase the directory didn't already exist
    os.makedirs(full_cam_dir,exist_ok=True)
    os.makedirs(full_lidar_dir,exist_ok=True)
    

    save_num = get_lastest_fnum() + 1
    print("First save at:\n\t{}\\lidar_{}.npy\n\t{}\\cam_{}.png".format(full_lidar_dir,save_num,full_cam_dir,save_num))
    print("Going into take_data function")
    take_data(full_lidar_dir,full_cam_dir,save_num)


if __name__ == "__main__":
    main()