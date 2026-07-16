import blickfeld_qb2
import numpy as np
from pylsl import StreamInfo, StreamOutlet
import threading
import time
import asyncio
from blickfeld_qb2.base.grpc.channel import Channel
import socket
import os

def record_qb2(file_prefix='qb2', device_name="192.168.50.36"):
    # LSL
    info = StreamInfo('blickfield_qb2', 'Image', 1, 8, 'int32', 'qb2-xxxx')
    outlet = StreamOutlet(info)

    channel = blickfeld_qb2.Channel(fqdn_or_ip=device_name)

    # client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # client.connect(('localhost', 9999))

    print("qb2 ready")

    out_dir = input("Press enter to continue with session ID")
    out_dir = os.path.join("/Volumes/T7", out_dir)
    os.makedirs(out_dir, exist_ok=True)

    with blickfeld_qb2.Channel(fqdn_or_ip=device_name) as channel:
        service = blickfeld_qb2.core_processing.services.PointCloud(channel)

        for i, response in enumerate(service.stream()):

            # Extract a point cloud frame from the response
            frame = response.frame

            outlet.push_sample([frame.id])

            # Print the frame ID
            print("Received frame with ID:", frame.id)

            # response = client.recv(1024).decode()
            # should_continue = response.lower() == 'true'
            # print(f"Should continue: {should_continue}")

            out_path = os.path.join(out_dir, f"{file_prefix}-{frame.id}.npy")
            with open(out_path, "wb") as f:
                np.save(f, frame, allow_pickle=True)

            # if i > 10:
            #     break


if __name__ == '__main__':
    record_qb2()