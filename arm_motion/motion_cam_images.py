import sys, os, cv2, tqdm
z1_sdk_dir = "/home/tomtang/Documents/droplab/z1_teaching" # directory containing z1_sdk
sys.path.append(z1_sdk_dir)
from z1_sdk.lib import unitree_arm_interface
import numpy as np

def grab_tool(arm, close_ang=0.174533, duration=20, jnt_speed=1.0):
    init_pose = [-0.00099, -0.06995, -0.00049, 0.08664, -0.00015, 0.17644]
    num_iter = int(duration/2)
    for i in tqdm.tqdm(range(num_iter), desc ="Opening gripper"):
        arm.MoveJ(init_pose, -np.pi/2, jnt_speed)
    for i in tqdm.tqdm(range(num_iter), desc ="Closing gripper"):
        arm.MoveJ(init_pose, -close_ang, jnt_speed)
    return close_ang

def release_tool(arm, duration=20, jnt_speed=1.0):
    init_pose = [-0.00099, -0.06995, -0.00049, 0.08664, -0.00015, 0.17644]
    num_iter = int(duration/2)
    for i in tqdm.tqdm(range(num_iter), desc ="Opening gripper"):
        arm.MoveJ(init_pose, -np.pi/2, jnt_speed)
    for i in tqdm.tqdm(range(num_iter), desc ="Closing gripper"):
        arm.MoveJ(init_pose, 0.0, jnt_speed)

# Open handles to the webcams

def generate_images(end_effector_angles, test_poses, tg_gripper_angs, conti_move_idxs,
                    duration=20, save_format='png', saving_dir='image_captured', cam_idx=2):
    #assert len(end_effector_angles) == len(tg_gripper_angs)
    print("Press ctrl+\ to quit process.")
    np.set_printoptions(precision=3, suppress=True)
    arm =  unitree_arm_interface.ArmInterface(hasGripper=True)
    # arm_model = arm._ctrlComp.armModel
    armState = unitree_arm_interface.ArmFSMState
    arm.loopOn()
    arm.backToStart()
    
    # grab_tool(arm, close_ang=tg_gripper_angs, duration=duration, jnt_speed=1.0)
    total_num_images = len(end_effector_angles)
    for i in range(total_num_images):
        print(f"Start Taking {i + 1} out of {total_num_images}")
        jnt_speed = 1.0
        
        tg_pose = np.array(end_effector_angles[i])
        arm.MoveJ(tg_pose, tg_gripper_angs, jnt_speed)
        cam = cv2.VideoCapture(cam_idx)
        cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, -4.25)  # Example value
        success_captured, img = cam.read()
        img = cv2.flip(img, -1) 
        if success_captured:
            saved_name = f'{i:02}.{save_format}'
            cv2.imwrite(f'{saving_dir}/train/{saved_name}', img)
            print(saved_name, "saved!")   
        else:
            print(f"unable to capture frame {i}")
        cam.release()
        if i not in conti_move_idxs or (i == total_num_images - 1):
            arm.backToStart()

    for i in range(len(test_poses)):
        print(f"Start Taking {i + 1} out of {len(test_poses)}")
        jnt_speed = 1.0
        
        tg_pose = np.array(test_poses[i])
        arm.MoveJ(tg_pose, tg_gripper_angs, jnt_speed)
        cam = cv2.VideoCapture(cam_idx)
        cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, -4.25)  # Example value
        success_captured, img = cam.read()
        img = cv2.flip(img, 0) 
        if success_captured:
            saved_name = f'{i}_test.{save_format}'
            cv2.imwrite(f'{saving_dir}/test/{saved_name}', img)
            print(saved_name, "saved!")   
        else:
            print(f"unable to capture frame {i}")
        cam.release()
        if (i == len(test_poses) - 1):
            arm.backToStart()
    # release_tool(arm, duration=20, jnt_speed=1.0)
    arm.loopOff()

    cv2.destroyAllWindows()