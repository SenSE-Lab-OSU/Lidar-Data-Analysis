import sys
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

def main():
    # 1. Check if the user provided a file path argument
    if len(sys.argv) < 2:
        print("Error: Please provide the path to the JSON file.")
        print("Usage: python readcalib.py \\path\\to\\file.json")
        sys.exit(1)

    file_path = Path(sys.argv[1])

    # 2. Open and parse the JSON file safely
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' could not be found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: '{file_path}' is not a valid JSON file.")
        sys.exit(1)

    # 3. Extract data from the JSON structure
    t = np.array(data["translation"])
    quat = data["quaternion"] 

    # 4. Process Rotation and build the 4x4 Transformation Matrix
    r = R.from_quat(quat)
    rot_matrix = r.as_matrix()
    rot_vec_deg = r.as_rotvec(degrees=True)

    # 5. Construct LaTeX Friendly Formats
    latex_T = f"""\\begin{{bmatrix}}
{rot_matrix[0][0]:.4f} & {rot_matrix[0][1]:.4f} & {rot_matrix[0][2]:.4f} & {t[0]:.4f} \\\\
{rot_matrix[1][0]:.4f} & {rot_matrix[1][1]:.4f} & {rot_matrix[1][2]:.4f} & {t[1]:.4f} \\\\
{rot_matrix[2][0]:.4f} & {rot_matrix[2][1]:.4f} & {rot_matrix[2][2]:.4f} & {t[2]:.4f} \\\\
0.0000 & 0.0000 & 0.0000 & 1.0000
\\end{{bmatrix}}"""

    latex_R = f"\\begin{{bmatrix}} {rot_vec_deg[0]:.4f}^\\circ & {rot_vec_deg[1]:.4f}^\\circ & {rot_vec_deg[2]:.4f}^\\circ \\end{{bmatrix}}"
    latex_t = f"\\begin{{bmatrix}} {t[0]:.4f} & {t[1]:.4f} & {t[2]:.4f} \\end{{bmatrix}}"

    # 6. Construct Plain Text Friendly Formats
    plain_T = f"""[{rot_matrix[0][0]:.4f}, {rot_matrix[0][1]:.4f}, {rot_matrix[0][2]:.4f}, {t[0]:.4f}]
[{rot_matrix[1][0]:.4f}, {rot_matrix[1][1]:.4f}, {rot_matrix[1][2]:.4f}, {t[1]:.4f}]
[{rot_matrix[2][0]:.4f}, {rot_matrix[2][1]:.4f}, {rot_matrix[2][2]:.4f}, {t[2]:.4f}]
[0.0000, 0.0000, 0.0000, 1.0000]"""

    plain_R = f"[{rot_vec_deg[0]:.4f}, {rot_vec_deg[1]:.4f}, {rot_vec_deg[2]:.4f}] degrees"
    plain_t = f"[{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]"

    # 7. Print outputs
    print("=================== LaTeX FORMAT ===================")
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
    main()