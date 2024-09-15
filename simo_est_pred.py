import torch, pickle, cv2, roma
import numpy as np
import matplotlib.pyplot as plt
from helper import project_points, diff_AB, intrin_to_krt
from pose_config import intrinsics

fx, fy, cx, cy = intrinsics
K, R, t = intrin_to_krt(fx, fy, cx, cy)

"""
Notice pred function when plot img/projection on matplotlib plot from the upper origin, so the
prediction looks upside-down
"""

def pred_on_train(poses_t, eef_poses, pts_tor, rgb_tor, num_used, T, scale_t, train_dir):
    for ii in range(num_used):
        pred_list=[]
        for ii_base in range(0,num_used):
            A_0_o = torch.cat((eef_poses[ii_base].reshape(1, -1, 4), eef_poses[ii].float().reshape(1, -1, 4)), dim=0)
            A_0,_ = diff_AB(A_0_o, A_0_o)
            A_0 = A_0.squeeze(0)
            B_0 = (torch.linalg.pinv(T).float())@A_0@T.float()
            B_0_s=B_0.clone()
            B_0_s[:3,3]=B_0[:3,3]/scale_t.cpu()
            pred_s = poses_t[ii_base]@B_0_s
            pred_list.append(pred_s.clone())
        pred_m=torch.stack(pred_list).mean(dim=0)
        pred=torch.eye(4)
        pred[:3,:3]=roma.special_procrustes(pred_m[:3,:3])
        pred[:3,3]=pred_m[:3,3]
        tool_3d_h=torch.cat([pts_tor,torch.ones(len(pts_tor),1)],dim=-1)
        transformed_point_homogeneous = tool_3d_h@pred.T
        points_3d_tools = transformed_point_homogeneous[:,:3]
        ptc_2d,depths_tool=project_points(points_3d_tools.float(), K, R, t)
        ptc_2d=ptc_2d.detach().cpu()
        #x, y, c = ptc_2d[::5,0],ptc_2d[::5,1],rgb_tor[::5]
        plt.figure()
        plt.imshow(cv2.imread(f"{train_dir}/masked_{ii:02}.png"),cmap='Reds',alpha=.5)
        plt.scatter(ptc_2d[::5,0],ptc_2d[::5,1],c=rgb_tor[::5].numpy(),s=2,alpha=0.5)
        plt.xlim([0,640])
        plt.ylim([0,480])
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.show()

def pred_on_test(R_c, trans_c, scale_t_c, 
                 test_poses, poses_t, eef_poses, pts_tor, rgb_tor, num_used, 
                 data_dir, tool, save_pred_imgs=False):
    test_dir = f"{data_dir}/{tool}/test"
    for ii in range(4):
        pred_list=[]
        for ii_base in range(0,num_used):
            A_0_o = torch.cat((eef_poses[ii_base].reshape(1, -1, 4), test_poses[ii].float().reshape(1, -1, 4)), dim=0)
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

        tool_3d_h=torch.cat([pts_tor,torch.ones(len(pts_tor),1)],dim=-1)
        transformed_point_homogeneous = tool_3d_h@pred.T
        points_3d_tools = transformed_point_homogeneous[:,:3]
        ptc_2d,depths_tool=project_points(points_3d_tools.float(), K, R, t)
        ptc_2d=ptc_2d.detach().cpu()

        plt.figure()
        ptc_2d=ptc_2d.detach().cpu()
        plt.imshow(cv2.imread(f"{test_dir}/maskof{ii}.png"))
        plt.scatter(ptc_2d[::1,0],ptc_2d[::1,1],s=2,c=rgb_tor[::1].numpy(),alpha=0.1)
        plt.xlim([0,640])
        plt.ylim([0,480])
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.show()

    img_gt_list=[]
    img_pred_list=[]
    for ii in range(4):
        pred_list=[]
        for ii_base in range(0,num_used):
            A_0_o = torch.cat((eef_poses[ii_base].reshape(1, -1, 4), test_poses[ii].float().reshape(1, -1, 4)), dim=0)
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
        
        tool_3d_h=torch.cat([pts_tor,torch.ones(len(pts_tor),1)],dim=-1)
        transformed_point_homogeneous = tool_3d_h@pred.T
        points_3d_tools = transformed_point_homogeneous[:,:3]
        ptc_2d, _=project_points(points_3d_tools.float(), K, R, t)
        ptc_2d=ptc_2d.detach().cpu()
        
        input_img=cv2.imread(f"{test_dir}/masked_{ii:02}.png")
        white_points = np.where(input_img == 255)
        val_points = np.column_stack((white_points[1], white_points[0]))
        ptc_sel_filter=(ptc_2d[:,0]>=0)&(ptc_2d[:,0]<=640)&(ptc_2d[:,1]>=0)&(ptc_2d[:,1]<=480)
        ptc_2d_sel=ptc_2d[ptc_sel_filter]
        img_pred_list.append(ptc_2d_sel)

        img_gt_list.append(val_points)

    if save_pred_imgs:
       
        obj_comp_pred_path = f"{data_dir}/{tool}/{tool}_{num_used}_pred.pkl"
        obj_comp_gt_path = f"{data_dir}/{tool}/{tool}_{num_used}_gt.pkl"
        with open(obj_comp_gt_path, 'wb') as file:
            pickle.dump(img_gt_list, file)
        with open(obj_comp_pred_path, 'wb') as file:
            pickle.dump(img_pred_list, file)


"""
init test
"""
def pred_on_test_init(T, scale_t, 
                      test_poses, poses_t, eef_poses, pts_tor, rgb_tor, num_used, 
                      data_dir, tool, save_pred_imgs=False):
    test_dir = f"{data_dir}/{tool}/test"
    img_init_list=[]
    for ii in range(4):
        pred_list=[]
        for ii_base in range(0,num_used):
            A_0_o = torch.cat((eef_poses[ii_base].reshape(1, -1, 4), test_poses[ii].float().reshape(1, -1, 4)), dim=0)
            A_0,_ = diff_AB(A_0_o, A_0_o)
            A_0 = A_0.squeeze(0)
            B_0 = (torch.linalg.pinv(T).float())@A_0@T.float()
            B_0_s=B_0.clone()
            B_0_s[:3,3]=B_0[:3,3]/scale_t.cpu()
            pred_s = poses_t[ii_base]@B_0_s
            pred_list.append(pred_s.clone())
        pred_m=torch.stack(pred_list).mean(dim=0)
        pred=torch.eye(4)
        pred[:3,:3]=roma.special_procrustes(pred_m[:3,:3])
        pred[:3,3]=pred_m[:3,3]
        tool_3d_h=torch.cat([pts_tor,torch.ones(len(pts_tor),1)],dim=-1)
        transformed_point_homogeneous = tool_3d_h@pred.T
        points_3d_tools = transformed_point_homogeneous[:,:3]
        ptc_2d, _=project_points(points_3d_tools.float(), K, R, t)
        ptc_2d=ptc_2d.detach().cpu()
        plt.figure()
        #plt.imshow(cv2.imread(f"brush_tr_sv_2/img{ii:02}.png"),cmap='Reds',alpha=.5)
        input_img=cv2.imread(f"{test_dir}/maskof{ii}.png")
        plt.imshow(input_img)
        plt.scatter(ptc_2d[::10,0],ptc_2d[::10,1],c=rgb_tor[::10].numpy(),s=2,alpha=0.5)
        plt.xlim([0,640])
        plt.ylim([0,480])
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.show()
        ptc_sel_filter=(ptc_2d[:,0]>=0)&(ptc_2d[:,0]<=640)&(ptc_2d[:,1]>=0)&(ptc_2d[:,1]<=480)
        ptc_2d_sel=ptc_2d[ptc_sel_filter]
        img_init_list.append(ptc_2d_sel)
    if save_pred_imgs:
        init_comp_path = f"{data_dir}/{tool}/init_comp.pkl"
        with open(init_comp_path, 'wb') as file:
            pickle.dump(img_init_list, file)