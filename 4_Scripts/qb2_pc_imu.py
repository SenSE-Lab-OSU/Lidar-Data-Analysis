# %%
import blickfeld_qb2
import numpy as np
from tqdm import tqdm


data_all = []
counter = 0

# Open a secure connection
with blickfeld_qb2.Channel(fqdn_or_ip="192.168.0.253") as channel:
    
    # Access the Acceleration service
    # This is located under core_processing.services

    for i in tqdm(range(100)):
        service = blickfeld_qb2.core_processing.services.Acceleration(channel)
        
        # print(dir(service))
        # print("Starting acceleration stream...")
        
        res = service.get_filtered()
        data = [res.acceleration.x, res.acceleration.y, res.acceleration.z]
        data_all.append(data)
    # print(dir(res.acceleration))
    # print(data)


# %%
data_all = np.stack(data_all)
np.save("./data/imu.npy", data_all)