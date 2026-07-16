import numpy as np
import blickfeld_qb2
import os
from pypcd4 import PointCloud

# Convert Lidar data saved as npy format to pcd file
# Using python 3.13, open3d only up to version 3.12
# Using pypcd package instead

print("Looking for .npy files in {}".format(os.getcwd()))
print("Found: ")
files = list(os.walk('.'))[0][2]
for file in files:
    fname = os.path.splitext(file)
    if fname[1] == ".npy":
        print("\t{}".format(file),end=" ")   
        try:
            lidarData = np.load(file, allow_pickle=True)
            lidarData = lidarData.tolist()
            lidarF = lidarData.binary.cartesian
            pc = PointCloud.from_xyz_points(lidarF)
            pc.save(fname[0]+".pcd")
            print("Saved")
        except Exception as e:
            print("Failed to save: ",end="")
            print(e)


# for i in range(num1,num2+1):
#     # loads lidar as blickfeld.core_processing.data.Frame
#     lidarData = np.load(dataName.format(i), allow_pickle=True)
#     lidarData = lidarData.tolist()[0]

#     # Extracts cartesian points
#     lidarF = lidarData.binary.cartesian
#     pc = PointCloud.from_xyz_points(lidarF)
#     pc.save(dataSave.format(i))
    
