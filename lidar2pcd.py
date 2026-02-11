import numpy as np
import blickfeld_qb2
from pypcd4 import PointCloud

# Convert Lidar data saved as npy format to pcd file
# Using python 3.13, open3d only up to version 3.12
# Using pypcd package instead
dataName = "data/lidar_{}.npy"
dataSave = "data/lidar_{}.pcd"
print("Using "+dataName)
num1 = int(input("File Start: "))
num2 = int(input("File end: "))

for i in range(num1,num2+1):
    # loads lidar as blickfeld.core_processing.data.Frame
    lidarData = np.load(dataName.format(i), allow_pickle=True)
    lidarData = lidarData.tolist()[0]

    # Extracts cartesian points
    lidarF = lidarData.binary.cartesian
    pc = PointCloud.from_xyz_points(lidarF)
    pc.save(dataSave.format(i))
