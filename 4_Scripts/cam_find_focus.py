"""
cam_find_focus.py


useage: python cam_find_focus.py --camera-index N
"""

# Significant speed improvement when set BEFORE importing cv2.
import os
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
import cv2 as cv
import numpy as np
import argparse
from time import sleep

# GLOBALS
DEFAULT_CAMERA_INDEX = 0 # Should be built-in cam if exists, or plugged in webcam, need to set --cam-index to 1 if you want to use webcam AND you have built-in 
DEFAULT_CAMERA_WIDTH = 3840
DEFAULT_CAMERA_HEIGHT = 2160

# Open Cam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture camera data and find max focus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=DEFAULT_CAMERA_INDEX,
        help="OpenCV camera index to open (0 is usually the default/built-in camera).",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Manually decreased/increase focus with '['/']'"
    )
    return parser.parse_args()
def capture_loop(cap: cv.VideoCapture, manual: bool) -> int:
    pos = [5,25]
    pos2 = pos.copy()
    font = cv.FONT_HERSHEY_COMPLEX_SMALL
    size = 1
    color = (255,255,255)
    thickness = 1
    line = cv.LINE_AA

    text_size, _ = cv.getTextSize(f"Set Focus 1, Reported Focus: 1", font, size, thickness)
    line_height = text_size[1]
    pos2[1] = line_height + pos2[1] + 5

    set_focus = cap.get(cv.CAP_PROP_FOCUS)
    print(f"Initial focus value at: {set_focus}")
    while True:
        cap.set(cv.CAP_PROP_FOCUS,set_focus)
        cur_focus = cap.get(cv.CAP_PROP_FOCUS)

        ret, frame = cap.read()
        cv.putText(frame,f"Set Focus {set_focus}, Reported Focus: {cur_focus}", pos,font,size,color,thickness,line)
        if manual: 
            cv.putText(frame, "Use '['/']' to manually increase decrease", pos2, font, size, color, thickness, line)
        cv.imshow('Camera',frame)
        key = cv.waitKey(1)
        if key == ord('[') and manual:
            set_focus = set_focus - 1
            print(f"Read decrease focus at {set_focus}")
        elif key == ord(']') and manual:
            set_focus = set_focus + 1
            print(f"Read increase focus at {set_focus}")
        elif key == ord('q') and manual:
            cv.destroyAllWindows()
            return cur_focus
            break
        if not manual and set_focus == cur_focus:
            sleep(2)
            print("Increasing Focus by 1")
            set_focus = set_focus + 1   
        elif not manual and set_focus > cur_focus:
            print(f"Set to {set_focus}, but only {cur_focus}, exiting")
            sleep(3)
            cv.destroyAllWindows()
            return cur_focus
    return 0

   


def main() -> None:
    args = parse_args()
    cap = cv.VideoCapture(args.camera_index)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, DEFAULT_CAMERA_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT,DEFAULT_CAMERA_HEIGHT)

    reported_w = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    reported_h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera using width x height: {reported_w} x {reported_h}")
    desired_focus = capture_loop(cap,args.manual)

    print(25*'_')
    print(f"The Final Focus was set to {desired_focus}")
    print(25*'_')



if __name__ == "__main__":
    main()