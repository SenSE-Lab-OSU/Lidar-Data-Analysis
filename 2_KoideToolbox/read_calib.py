import json
import argparse
from scipy.spatial.transform import Rotation as R
import numpy as np


def print_Tform(r_quat, t_vec):
    # 1. Invert the transform logic matching your original script
    r = R.from_quat(r_quat)
    rot = r.inv().as_matrix()
    rotvec = r.inv().as_rotvec(degrees=True)
    t = -r.apply(t_vec, inverse=True)

    # 2. Construct LaTeX Friendly Formats
    latex_T = f"""\\begin{{bmatrix}}
{rot[0][0]:.4f} & {rot[0][1]:.4f} & {rot[0][2]:.4f} & {t[0]:.4f} \\\\
{rot[1][0]:.4f} & {rot[1][1]:.4f} & {rot[1][2]:.4f} & {t[1]:.4f} \\\\
{rot[2][0]:.4f} & {rot[2][1]:.4f} & {rot[2][2]:.4f} & {t[2]:.4f} \\\\
0.0000 & 0.0000 & 0.0000 & 1.0000
\\end{{bmatrix}}"""

    latex_R = f"{rotvec[0]:.4f}^\\circ & {rotvec[1]:.4f}^\\circ & {rotvec[2]:.4f}^\\circ"
    latex_t = f"{t[0]:.4f} & {t[1]:.4f} & {t[2]:.4f}"

    # 3. Construct Plain Text Friendly Formats
    plain_T = f"""[{rot[0][0]:.4f}, {rot[0][1]:.4f}, {rot[0][2]:.4f}, {t[0]:.4f}]
[{rot[1][0]:.4f}, {rot[1][1]:.4f}, {rot[1][2]:.4f}, {t[1]:.4f}]
[{rot[2][0]:.4f}, {rot[2][1]:.4f}, {rot[2][2]:.4f}, {t[2]:.4f}]
[0.0000, 0.0000, 0.0000, 1.0000]"""

    plain_R = f"[{rotvec[0]:.4f}, {rotvec[1]:.4f}, {rotvec[2]:.4f}] degrees"
    plain_t = f"[{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]"

    # 4. Print outputs clearly split by format
    print("=================== LATEX FORMAT ===================")
    print("T =")
    print(latex_T)
    print("\nR =")
    print(latex_R)
    print("\nt =")
    print(latex_t)
    
    print("\n================= PLAIN TEXT FORMAT =================")
    print("T =")
    print(plain_T)
    print("\nR =")
    print(plain_R)
    print("\nt =")
    print(plain_t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read Calib.json file and print transform")
    parser.add_argument("DIR", type=str, help="The target JSON file path.")
    args = parser.parse_args()
    
    with open(args.DIR, "r") as f:
        calib = json.load(f)
        
    try:
        T = calib['results']['T_lidar_camera']
        print("Read Calibration successfully.\n")
    except KeyError:
        T = calib['results']['init_T_lidar_camera']
        print("Failed to read T_lidar_camera, defaulted to initial transform.\n")
        
    t_vec = T[:3]
    r_quat = T[3:]
    
    print_Tform(r_quat, t_vec)