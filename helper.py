import torch, pickle, os, sys, cv2, tqdm, roma
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import open3d as o3d
from scipy.spatial.transform import Rotation as R_mod
from PIL import Image

def project_points(points_3d, K, R, t):
    points_cam = torch.mm(points_3d, R.t()) + t
    points_cam_homogeneous = torch.cat((points_cam, 
                                        torch.ones(points_cam.shape[0], 
                                                   1, device=points_cam.device)), dim=1)
    points_homogeneous = torch.mm(points_cam_homogeneous[:,:3], K.t())
    points = points_homogeneous[:,:2]/ points_homogeneous[:, 2:]
    depth = points_homogeneous[:, 2:]
    return points, depth

def normalize_vector(v):
    """ Normalize a vector. """
    norm = torch.norm(v)
    if norm == 0:
        raise ValueError("Zero vector cannot be normalized")
    return v / norm

def orthogonalize_vectors(tangent, bitangent, normal):
    tangent = normalize_vector(tangent)
    bitangent = bitangent - torch.dot(bitangent, tangent) * tangent
    bitangent = normalize_vector(bitangent)
    normal = normal - torch.dot(normal, tangent) * tangent
    normal = normal - torch.dot(normal, bitangent) * bitangent
    normal = normalize_vector(normal)
    return tangent, bitangent, normal

def pose_from_vectors(position, normal, tangent, bitangent):
    #tangent, bitangent, normal = orthogonalize_vectors(tangent, bitangent, normal)
    rotation_matrix = torch.stack([normal, tangent, bitangent], dim=1)
    
    if rotation_matrix.shape[0] != 3 or rotation_matrix.shape[1] != 3:
        raise ValueError("Vectors must be 3-dimensional")
    
    transform_matrix = torch.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = position
    return transform_matrix

def diff_AB(eff_poses_tor,im_poses):
    A, B = [], []
    for i in range(1,len(im_poses)):
        p = eff_poses_tor[i-1], im_poses[i-1]
        n = eff_poses_tor[i], im_poses[i]
        A.append(torch.mm(torch.linalg.inv(p[0]), n[0]))
        B.append(torch.mm(torch.linalg.inv(p[1]), n[1]))
    A_all = torch.stack(A)
    B_all = torch.stack(B)
    return A_all, B_all
def diff_AB_fully_connected(eff_poses_tor, im_poses):
    # Move tensors to GPU if available
    eff_poses_tor = eff_poses_tor.to('cuda')
    im_poses = im_poses.to('cuda')
    n = eff_poses_tor.shape[0]
    # Create indices for the i and j combinations
    indices = torch.tensor([(i, j) for i in range(n) for j in range(n) if i != j], device='cuda')
    i_indices = indices[:, 0]
    j_indices = indices[:, 1]
    # Gather the necessary pairs
    p_eff = eff_poses_tor[i_indices]
    n_eff = eff_poses_tor[j_indices]
    p_im = im_poses[i_indices]
    n_im = im_poses[j_indices]
    # Calculate the pseudo-inverse
    p_eff_pinv = torch.linalg.pinv(p_eff)
    p_im_pinv = torch.linalg.pinv(p_im)
    # Perform batch matrix multiplication
    A_all = torch.bmm(p_eff_pinv, n_eff)
    B_all = torch.bmm(p_im_pinv, n_im)
    labels = indices.tolist()
    
    return A_all, B_all, labels
def combine_matrices(rot_matrix, translation_matrix):
    combined_matrix = np.eye(4)
    combined_matrix[:3, :3] = rot_matrix
    combined_matrix[:3, 3] = translation_matrix.squeeze()
    
    return combined_matrix

def to_tensor(cm):
    cf_list=[]
    for i in range(len(cm)):
        cf=torch.tensor(cm[i]).detach()
        cf_list.append(cf)
    cm=torch.stack(cf_list)
    return(cm)

def rpy_to_rot_matrix(ori):
    r = R_mod.from_euler('xyz', ori, degrees=False)
    return torch.tensor(r.as_matrix())

def pose_to_transform(pose_batch):
    # Unpack the pose components
    pos = pose_batch[:, 3:6]

    # Convert RPY to rotation matrices
    rotation_matrices = rpy_to_rot_matrix(pose_batch[:, :3])

    # Create the transformation matrices
    transform_matrices = torch.zeros((pose_batch.shape[0], 4, 4), dtype=torch.float)
    transform_matrices[:, :3, :3] = rotation_matrices
    transform_matrices[:, :3, 3] = pos
    transform_matrices[:, 3, 3] = 1.0
    return transform_matrices

def intrin_to_krt(fx, fy, cx, cy):
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0., 1.]]).float()
    R = torch.eye(3)
    t = torch.tensor([0.,0.,0.])
    return K, R, t

def read_preprocessed(data_dir, tool, num_used):
    preprocess_path = f"{data_dir}/{tool}/train_preprocessed_{num_used}.tar"
    save_dict = torch.load(preprocess_path,map_location=torch.device('cpu'))
    imgs=save_dict["imgs"]
    poses=save_dict["poses"]
    pts3d=save_dict["pts3d"]
    confidence_masks=save_dict["confidence_masks"]
    masks=save_dict["seg_mask"]

    pts3d_np_te = to_tensor(pts3d)
    imgs_te=to_tensor(imgs)
    pts_te = np.concatenate([p[m] for p, m in zip(pts3d_np_te[:], masks[:].bool())])
    rgb_colors_te = np.concatenate([p[m] for p, m in zip(imgs_te[:], masks[:].bool())])
    pts_tor=torch.tensor(pts_te)
    rgb_tor=torch.tensor(rgb_colors_te)
    return poses, pts_tor, rgb_tor, masks

def viz_masks_img_from_file(masks, imgs_paths, n=9):
    for i in range(n):
        image_pil = Image.open(imgs_paths[i]).convert("RGB")
        tor_img=torch.tensor(np.array(image_pil))
        plt.imshow(tor_img)
        plt.imshow(masks[i].numpy(), cmap='plasma', alpha=0.5)
        plt.axis('off')  # Hide axes
        plt.show()

def viz_imgs(imgs, num_used=9):
    for i in range(num_used):
        plt.imshow(imgs[i])
        plt.axis('off')  
        plt.show()

def viz_rgb_ptc(xyz, rgb):
    if not isinstance(xyz, np.ndarray):
        if isinstance(xyz, torch.Tensor):
            xyz = xyz.cpu().numpy()
        else:
            xyz = np.array(xyz)

    if not isinstance(rgb, np.ndarray):
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.cpu().numpy()
        else:
            rgb = np.array(rgb)


    point_cloud = o3d.geometry.PointCloud()

    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    point_cloud.colors = o3d.utility.Vector3dVector(rgb)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud])