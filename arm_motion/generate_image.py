import os, sys
from motion_cam_images import generate_images
sys.path.append('../')
from pose_config import side_cam_cposes, test_poses, train_poses_idx


tool = "screw_driver"
saving_dir = f"../data/{tool}"
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)
train_dir = f"{saving_dir}/train"
test_dir = f"{saving_dir}/test"
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
generate_images(end_effector_angles=side_cam_cposes[tool], 
                test_poses=test_poses, 
                tg_gripper_angs=0.02, conti_move_idxs=list(range(len(side_cam_cposes[tool]))),
                    duration=30, save_format='jpg', saving_dir=saving_dir, cam_idx=2)