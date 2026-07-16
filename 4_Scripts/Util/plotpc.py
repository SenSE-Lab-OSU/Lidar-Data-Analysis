import matplotlib
import json
import os
import numpy as np
from pypcd4 import PointCloud

# Get File location
config = "./config.json"
with open(config) as f:
    fpath = json.load(f)['dataPath']
print(fpath)
folders = os.listdir(fpath)
for i,j in enumerate(folders):
    print("{}: {}".format(i,j))
# Assuming user will press the right button
foldersel = int(input("Select folder to use: "))
# Working directory
work = os.path.join(fpath,folders[foldersel])
print(work)

files = os.listdir(work)


for file in files:
    front, ext = os.path.splitext(file)
    if ext == ".npy":
        data = np.load(os.path.join(work,file))
        pc = PointCloud.from_xyz_points(data)
        pc.save(os.path.join(work,front+".pcd"))