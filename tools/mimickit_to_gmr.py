"""
Convert MimicKit humanoid motion (.pkl) to GMR-format .pkl.

Input (MimicKit):
    Pickle dict with:
        - "loop_mode": int
        - "fps": float or int
        - "frames": list/array of shape (T, 34)
          frame layout for humanoid.xml:
            [root_pos(3),
             root_rot_expmap(3),
             abdomen(3), neck(3),
             right_shoulder(3), right_elbow(1),
             left_shoulder(3),  left_elbow(1),
             right_hip(3),      right_knee(1), right_ankle(3),
             left_hip(3),       left_knee(1),  left_ankle(3)]

Output (GMR):
    Pickle dict with:
        - "fps": fps
        - "root_pos": (T, 3)
        - "root_rot": (T, 4)  quaternion (x, y, z, w)
        - "dof_pos":  (T, 28) all remaining DOFs
        - "local_body_pos": None
        - "link_body_list": None
"""

import argparse
import pickle
import numpy as np
import torch

# You already have expmap->quat in MimicKit; we need quat->expmap inverse here.
# GMR expects quaternions, so we implement expmap->quat.
# This is a standard axis-angle (Rodrigues) to quaternion conversion.

def expmap_to_quat(exp):
    """
    Convert exponential map (axis-angle, R^3) to quaternion (x, y, z, w).

    exp: (..., 3) tensor/ndarray
    return: (..., 4) tensor (x, y, z, w)
    """
    if isinstance(exp, np.ndarray):
        exp_t = torch.from_numpy(exp)
    else:
        exp_t = exp

    theta = torch.linalg.norm(exp_t, dim=-1, keepdim=True)  # (..., 1)
    # Avoid division by zero
    small = theta < 1e-8
    axis = torch.where(small, torch.zeros_like(exp_t), exp_t / theta.clamp_min(1e-8))
    half_theta = 0.5 * theta
    sin_half = torch.sin(half_theta)
    cos_half = torch.cos(half_theta)

    # axis-angle to quaternion: (x, y, z, w)
    quat_xyzw = torch.cat([axis * sin_half, cos_half], dim=-1)  # (..., 4)
    # For very small angles, approximate as identity
    quat_xyzw = torch.where(
        small.expand_as(quat_xyzw),
        torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=quat_xyzw.dtype, device=quat_xyzw.device),
        quat_xyzw,
    )
    return quat_xyzw


def convert_mimickit_to_gmr_humanoid(input_file, output_file):
    # Load MimicKit motion
    with open(input_file, "rb") as f:
        data = pickle.load(f)

    fps = data["fps"]
    frames = np.asarray(data["frames"], dtype=np.float32)  # (T, 34)
    T, D = frames.shape

    # Sanity check on dimensionality
    if D != 34:
        raise ValueError(
            f"Expected 34 DOFs per frame for humanoid.xml, got {D}. "
            "Make sure this script is only used for the humanoid character."
        )

    # Split into root position, root rotation (expmap), and remaining DOFs
    root_pos = frames[:, 0:3]          # (T, 3)
    root_rot_exp = frames[:, 3:6]      # (T, 3)
    dof_pos = frames[:, 6:]            # (T, 28)

    # Convert root rotation expmap -> quaternion (x,y,z,w)
    root_rot_quat = expmap_to_quat(root_rot_exp).cpu().numpy()  # (T, 4)

    # Pack into GMR format dict
    gmr_dict = {
        "fps": float(fps),
        "root_pos": root_pos,         # (T, 3)
        "root_rot": root_rot_quat,    # (T, 4), (x,y,z,w)
        "dof_pos": dof_pos,           # (T, 28)
        "local_body_pos": None,
        "link_body_list": None,
    }

    with open(output_file, "wb") as f:
        pickle.dump(gmr_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("=== MimicKit → GMR (humanoid) conversion complete ===")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"Frames: {T}, fps: {fps}")
    print("root_pos shape:", root_pos.shape)
    print("root_rot shape:", root_rot_quat.shape)
    print("dof_pos  shape:", dof_pos.shape)


def main():
    parser = argparse.ArgumentParser(
        description="Convert MimicKit humanoid motion (.pkl) to GMR-format .pkl."
    )
    parser.add_argument("--input_file", required=True, help="Path to MimicKit .pkl")
    parser.add_argument("--output_file", required=True, help="Path to output GMR .pkl")
    args = parser.parse_args()

    convert_mimickit_to_gmr_humanoid(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
