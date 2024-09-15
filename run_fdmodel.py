import torch
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from PIL import Image
import torchvision.transforms as transforms

path = os.path.join(os.getcwd(), "dust3r")
sys.path.append(path)

from dust3r.utils.device import to_numpy
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo,inf
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import plotly.graph_objects as go

def run_fdmodel(tool="screw_driver", num_img=3):
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 1500
    data_dir = 'data'
    train_processed_save_path = f"{data_dir}/{tool}/train_preprocessed_{num_img}.tar"
    train_directory=f"{data_dir}/{tool}/train"
    mask_tor_path = f"{train_directory}/mask_tor.pt"
    model_path = "weights/DUSt3R_ViTLarge_BaseDecoder_512_linear.pth"

    def load_model(model_path, device):
        print('... loading model from', model_path)
        ckpt = torch.load(model_path, map_location='cpu')
        args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
        if 'landscape_only' not in args:
            args = args[:-1] + ', landscape_only=False)'
        else:
            args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
        assert "landscape_only=False" in args
        print(f"instantiating : {args}")
        net = eval(args)
        print(net.load_state_dict(ckpt['model'], strict=False))
        return net.to(device)
    model=load_model(model_path,device)

    file_names_tr_dus = os.listdir(train_directory)
    file_names_tr_dus = [os.path.join(train_directory, file) for file in file_names_tr_dus if 'masked_' in file]
    file_names_tr_dus.sort()
    images_dust = load_images(file_names_tr_dus[:num_img], size=512)
    mask_tor = torch.load(mask_tor_path)[:num_img]
    pairs = make_pairs(images_dust, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer) #, bin_masks=mask_tor
    scene.preset_focal([580.0]*len(mask_tor))
    loss = scene.compute_global_alignment(init="mst", niter=niter, niter_PnP=1000, schedule=schedule, lr=lr)

    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.im_conf
    save_dict={"imgs":imgs, "focals":focals,"poses":poses,"pts3d":pts3d,"confidence_masks":confidence_masks,"seg_mask":mask_tor}
    torch.save(save_dict,train_processed_save_path) 
    return imgs, focals, poses, pts3d, confidence_masks, mask_tor

if __name__ == "__main__":
    run_fdmodel(tool='screw_driver', num_img=6)