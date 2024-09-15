import torch, tqdm, roma
import numpy as np
import matplotlib.pyplot as plt
from helper import (project_points, diff_AB, diff_AB_fully_connected, 
                    pose_to_transform, intrin_to_krt, read_preprocessed)
from pose_config import side_cam_cposes, test_poses, intrinsics
from simo_est_pred import pred_on_train, pred_on_test_init, pred_on_test
"""
set up camera intrinsics in pose_config.py
"""
fx, fy, cx, cy = intrinsics
K, R, t = intrin_to_krt(fx, fy, cx, cy)



def stage_1_opt(poses, pts_tor, masks, tool, num_used):    
    used_poses = side_cam_cposes[tool]
    eef_poses = pose_to_transform(torch.tensor(used_poses[:num_used]))
    
    poses_t = torch.linalg.pinv(poses).detach().cpu()

    print(f"Original ptc shape: {pts_tor.shape}")

    gt_inview=[]
    gt_bool=[]
    for ii in range(len(eef_poses)):
        tool_3d_h=torch.cat([pts_tor,torch.ones(len(pts_tor),1)],dim=-1)
        transformed_point_homogeneous = tool_3d_h@poses_t[ii].T
        points_3d_tools = transformed_point_homogeneous[:,:3] 

        masks_gt=(masks.permute((0,2,1)))
        ptc_2d_all, _=project_points(points_3d_tools.float(), K, R, t)
        sel_bool=(ptc_2d_all[:,0]>=0) & (ptc_2d_all[:,0]<=640) & (ptc_2d_all[:,1]>=0) & (ptc_2d_all[:,1]<=480)
        ptc_2d=ptc_2d_all[sel_bool]
        gt_inview.append(ptc_2d)
        gt_bool.append(sel_bool)

    for i in range(len(gt_inview)):
        cur_gt=gt_inview[i]
        plt.scatter(cur_gt[::20,0],cur_gt[::20,1])
    plt.show()

    in_A = eef_poses.clone()#[:-1]
    #in_A = in_A.requires_grad_()

    in_B = poses_t.clone()
    #in_B = in_B.requires_grad_()

    scale_t = torch.tensor([1.], requires_grad=True,device="cuda")
    trans = torch.zeros(3)
    trans = trans.requires_grad_()

    T_x_init = roma.random_rotmat(1)
    T_x = T_x_init.clone().cuda()
    T_x = T_x.requires_grad_()
    opt_T=torch.optim.Adam([{"params":T_x, "lr":0.003},
                            {"params":scale_t, "lr":0.001},
                        {"params":trans, "lr":0.003}
                        ])

    pbar = tqdm.tqdm(range(2000), desc='GD Caliberation')
    A, B, _= diff_AB_fully_connected(in_A, in_B)
    A=A.detach()
    B=B.detach()
    A=A.cuda()
    B=B.cuda()
    for ii in pbar:
        opt_T.zero_grad()
        T_x_s=roma.special_procrustes(T_x,gradient_eps=1e-04)
        
        T_x_s_aug=torch.eye(4,device="cuda")
        T_x_s_aug[:3,:3]=T_x_s[0]
        T_x_s_aug[:3,3]=trans
        
        B_scale=torch.eye(4,device="cuda").unsqueeze(0).repeat((len(B),1,1))
        B_scale[:,:3,:3]=B[:,:3,:3]
        B_scale[:,:3,3]=B[:,:3,3]*scale_t
        
        lhs=torch.bmm(torch.linalg.pinv(T_x_s_aug.unsqueeze(0).repeat((len(A),1,1))),
                    torch.bmm(A,T_x_s_aug.unsqueeze(0).repeat((len(A),1,1))))
        rhs=B_scale
        loss_R=torch.mean(roma.utils.rotmat_geodesic_distance(lhs[:, :3, :3], rhs[:, :3, :3]))
        #loss_reg_t=torch.norm(poses_t[:, :3, 3]-in_B[:, :3, 3],dim=-1).mean(dim=0)
        loss_t=(torch.norm(lhs[:, :3,3]-rhs[:, :3, 3],dim=-1)).mean(dim=0)
        #loss_reg = reg_factor * (loss_reg_R + loss_reg_t)
        loss = loss_R + loss_t# + loss_reg

        if(ii%50==0):
            pbar.set_description(f"{ii}, Loss R: {loss_R}, Loss t: {loss_t}, scale={float(scale_t[0])}")
        loss.backward()
        opt_T.step()
    T = T_x_s_aug.detach().cpu()
    return T, T_x_s_aug, scale_t, trans, poses_t, eef_poses, scale_t, gt_inview, gt_bool
    
def stage_2_opt(T_x_s_aug, scale_t, trans, poses_t, eef_poses, 
                pts_tor, gt_inview, gt_bool, num_used):

    tool_3d_h=torch.cat([pts_tor,torch.ones(len(pts_tor),1)],dim=-1).detach()
    R_c = T_x_s_aug[:3,:3].clone().cpu().detach()
    R_c.requires_grad_()
    scale_t_c=scale_t.clone().cpu().detach()
    scale_t_c.requires_grad_()
    trans_c=trans.clone().cpu().detach()
    trans_c.requires_grad_()
    opt_T=torch.optim.Adam([{"params":R_c, "lr":0.003},
                            {"params":scale_t_c, "lr":0.001},
                        {"params":trans_c, "lr":0.003}
                        ])
    max_iter=500
    for iterz in range(max_iter):
        opt_T.zero_grad()
        loss=0
        for ii in range(0,num_used):
            pred_list=[]
            for ii_base in range(0,num_used):
                A_0_o = torch.cat((eef_poses[ii_base].reshape(1, -1, 4), eef_poses[ii].float().reshape(1, -1, 4)), dim=0)
                A_0,_ = diff_AB(A_0_o, A_0_o)
                A_0 = A_0.squeeze(0)
            
                R_c_p=roma.special_procrustes(R_c,gradient_eps=1e-04)
                T_c = torch.eye(4)
                T_c[:3,:3]=R_c_p
                T_c[:3,3]=trans_c
                
                B_0 = (torch.linalg.pinv(T_c).float())@A_0@T_c.float()
                B_0_s=B_0.clone()
                B_0_s[:3,3]=B_0[:3,3]/scale_t_c.cpu()
                pred_s = poses_t[ii_base]@B_0_s
                pred_list.append(pred_s.clone())
            pred_m=torch.stack(pred_list).mean(dim=0)
            pred=torch.eye(4)
            pred[:3,:3]=roma.special_procrustes(pred_m[:3,:3])
            pred[:3,3]=pred_m[:3,3]
            cur_bool=gt_bool[ii]
            tool_3d_sel=tool_3d_h[cur_bool]
            sel_gt=gt_inview[ii]
            
            transformed_point_homogeneous = tool_3d_sel@pred.T
            points_3d_tools = transformed_point_homogeneous[:,:3]
            ptc_2d, _=project_points(points_3d_tools.float(), K, R, t)
            loss_cur=torch.norm(ptc_2d-sel_gt,dim=-1).mean()
            loss=loss+loss_cur
        if(iterz%50==0):
            print("{}:{}".format(iterz,loss))
        loss.backward()
        opt_T.step()
    return R_c, trans_c, scale_t_c

    
if __name__ == "__main__":
    # torch.manual_seed(3407) # from arxiv 2109.08203
    # np.random.seed(3407)
    tool = 'screw_driver'
    data_dir = 'data'
    num_used = 9
    
    test_poses = pose_to_transform(torch.tensor(test_poses).clone()).float()
    train_dir = f"{data_dir}/{tool}/train"

    poses, pts_tor, rgb_tor, masks = read_preprocessed(data_dir, tool, num_used)
    
    """
    Notice pred function when plot img/projection on matplotlib plot from the upper origin, so the
    prediction looks upside-down
    """
    print("done pred train")
    while (True):
        try:
            stage_1_res = stage_1_opt(poses, pts_tor, masks, tool, num_used)
            T, T_x_s_aug, scale_t, trans, poses_t, eef_poses, scale_t, gt_inview, gt_bool = stage_1_res
            # pred_on_train(poses_t, eef_poses, pts_tor, rgb_tor, num_used, T, scale_t, train_dir)
            pred_on_test_init(T, scale_t, test_poses, poses_t, eef_poses, pts_tor, rgb_tor, num_used, 
                      data_dir, tool, save_pred_imgs=True)
            print("done pred test init")
            R_c, trans_c, scale_t_c = stage_2_opt(T_x_s_aug, scale_t, trans, poses_t, eef_poses, 
                                                pts_tor, gt_inview, gt_bool, num_used)
            break
        except:
            print("")
            user_input = input("Gradient exceeding, retrying? (0 for quit, 1 for retrying)")
            if user_input == '0':
                print("Exiting")
                break
            else:
                print("Retrying...")
    print("Start final pred")
    pred_on_test(R_c, trans_c, scale_t_c, 
                 test_poses, poses_t, eef_poses, pts_tor, rgb_tor, num_used, 
                 data_dir, tool, save_pred_imgs=True)