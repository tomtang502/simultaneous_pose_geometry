import torch, os
import numpy as np
from helper import to_tensor, viz_rgb_ptc, viz_imgs
from pose_config import intrinsics
fx, fy, cx, cy = intrinsics
def run_viz_tool_only(tool, data_dir='data', save_dir='viz_tar', tool_tip_only=False, 
                      visualize_imgs=False):
    file = f'train_preprocessed_3.tar'
    if tool_tip_only:
        masks = torch.stack(torch.load(f"{data_dir}/{tool}/{tool}_tip_masks.pt"))
    else:
        masks = torch.stack(torch.load(f"{data_dir}/{tool}/{tool}_masks.pt"))

    preprocess_path = f"{data_dir}/{tool}/{file}"
    #print(masks)
    save_dict=torch.load(preprocess_path,map_location=torch.device('cpu'))
    imgs=save_dict["imgs"]
    num_used = len(imgs)
    pts3d=save_dict["pts3d"]

    pts3d_np_te = to_tensor(pts3d)
    imgs_te=to_tensor(imgs)

    pts_te = np.concatenate([p[m] for p, m in zip(pts3d_np_te[:], masks[:].bool())])
    rgb_colors_te = np.concatenate([p[m] for p, m in zip(imgs_te[:], masks[:].bool())])

    pts_tor=torch.tensor(pts_te)
    rgb_tor=torch.tensor(rgb_colors_te)
    print(f"Original points cloud shape: {pts_tor.shape} rgb shape: {rgb_tor.shape}")
    print(f"num image used: {num_used}, tool: {tool}")
    if visualize_imgs:
        viz_imgs(imgs, num_used)
    viz_rgb_ptc(pts_tor, rgb_tor)
    save_dict = {
        'ptc' : pts_tor,
        'rgb' : rgb_tor
    }
    if tool_tip_only:
        torch.save(save_dict, f"{save_dir}/{tool}_tip_ptc_rgb_dump.pth")
    else:
        torch.save(save_dict, f"{save_dir}/{tool}_only_ptc_rgb_dump.pth")
    return pts_tor, rgb_tor

if __name__ == "__main__":
    data_dir = 'data'
    save_dir = 'viz_tar'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    run_viz_tool_only('screw_driver', data_dir=data_dir, save_dir=save_dir, tool_tip_only=True, 
                      visualize_imgs=True)