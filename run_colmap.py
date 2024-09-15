import pycolmap, torch
import pathlib, os#, torch
import numpy as np
import open3d as o3d
import shutil, re
# import utils.geometric_util as geomu

tool = 'screw_driver'
num_img = 9
data_dir = 'data'
seg_mode = True
train_processed_save_path = f"{data_dir}/{tool}/train_preprocessed_{num_img}.tar"
train_directory=f"{data_dir}/{tool}/train"
if seg_mode:
    colmap_dir = f"{data_dir}/{tool}/colmap_seg"
else:
    colmap_dir = f"{data_dir}/{tool}/colmap"
mask_tor_path = f"{train_directory}/mask_tor.pt"

def is_seg_images(file):
    file = file.split('/')[-1]
    is_img_format = ('.png' in file) or ('.jpg' in file)
    return 'masked_' in file and is_img_format

def is_original_images(file):
    file = file.split('/')[-1]
    pattern = re.compile(r'\d{2}\.(jpg|png)')
    return pattern.fullmatch(file)


def copy_images_to_tmp(original_folder, n_imgs, parent_folder='tmp_dir', predicate=is_seg_images):
    """
    Copy specified images from the original folder to a temporary folder under the specified parent folder.

    Args:
    original_folder: Path to the original folder containing the images.
    parent_folder: Path to the parent folder where the temporary folder should be created.

    Returns:
    Path to the temporary folder where the images are copied.
    """

    # Create a temporary directory under the parent folder
    tmp_folder = os.path.join(parent_folder, "tmp")
    os.makedirs(tmp_folder, exist_ok=True)

    cnt = 0
    filenames = os.listdir(original_folder)
    filenames.sort()
    # Copy images to the temporary folder
    for filename in filenames:
        if (filename.endswith(('.jpg', '.jpeg', '.png', '.gif'))):
            original_path = os.path.join(original_folder, filename)
            if os.path.isfile(original_path) and predicate(original_path):
                shutil.copy(original_path, tmp_folder)
                print(original_path)
                cnt += 1
        if cnt >= n_imgs:
            break

    return tmp_folder

def delete_tmp_folder(tmp_folder):
    """
    Delete the temporary folder.

    Args:
    tmp_folder: Path to the temporary folder to delete.
    """
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)

def colmap_pose2transmat(col_pose_mat):
    return np.vstack((col_pose_mat, np.array([[0,0,0,1]])))


# delete_tmp_folder(colmap_dir)

output_path = pathlib.Path(colmap_dir)
print("Colmap saving folder at", output_path)
if not os.path.exists(output_path):
    original_folder = train_directory
    if seg_mode:
        tmp_folder = copy_images_to_tmp(original_folder, n_imgs=num_img, predicate=is_seg_images)
    else:
        tmp_folder = copy_images_to_tmp(original_folder, n_imgs=num_img, predicate=is_original_images)
    # Copy images to the temporary folder under the parent folder
    print("Images copied to temporary folder:", tmp_folder)
    print(output_path)
    image_dir = pathlib.Path(tmp_folder)

    output_path.mkdir()
    mvs_path = output_path / "mvs"
    database_path = output_path / "database.db"

    pycolmap.extract_features(database_path, image_dir)#, sift_options={"max_num_features": 512})
    #pycolmap.extract_features(database_path, image_dir)
    pycolmap.match_exhaustive(database_path)
    maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
    maps[0].write(output_path)

    pycolmap.undistort_images(mvs_path, output_path, image_dir)
    pycolmap.patch_match_stereo(mvs_path)  # requires compilation with CUDA
    pycolmap.stereo_fusion(output_path / "dense.ply", mvs_path)

    # Delete the temporary folder
    delete_tmp_folder(tmp_folder)
    print("Temporary folder deleted.")

reconstruction = pycolmap.Reconstruction(output_path)

col_cam_poses = []
selected_idx = []
idx_map = dict()
col_cam_poses_map = dict()
i = 0

if seg_mode:
    for image_id, image in reconstruction.images.items():
        name = image.name[:-len('.jpg')][len('masked_'):]
        print(name)
        idx = int(name)
        img_pose = np.array(image.cam_from_world.matrix())
        pose_tmat = torch.tensor(colmap_pose2transmat(img_pose))
        col_cam_poses.append(pose_tmat)
        col_cam_poses_map[idx] = pose_tmat.clone()
else:
    for image_id, image in reconstruction.images.items():
        name = image.name[:-len('.jpg')]
        print(name)
        idx = int(name)
        img_pose = np.array(image.cam_from_world.matrix())
        pose_tmat = torch.tensor(colmap_pose2transmat(img_pose))
        col_cam_poses.append(pose_tmat)
        col_cam_poses_map[idx] = pose_tmat.clone()

ply_path = os.path.join(colmap_dir, "dense.ply")
point_cloud = o3d.io.read_point_cloud(ply_path)

o3d.visualization.draw_geometries([point_cloud])