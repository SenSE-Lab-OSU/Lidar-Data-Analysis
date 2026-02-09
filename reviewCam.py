import numpy as np
import blickfeld_qb2


# Loads camera image as np.ndarray
camF = np.load("data/cam.npy")

# loads lidar as blickfeld.core_processing.data.Frame
lidarData = np.load("data/lidar.npy", allow_pickle=True)
lidarData = lidarData.tolist()[0]
# Extracts cartesian points
lidarF = lidarData.binary.cartesian

# Print some Frame data
print("Camera frame size: {}".format(camF.shape))
print("Lidar frame size: {}".format(lidarF.shape))

# Display camera image

# Display lidar image