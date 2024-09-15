import torch
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from lang_sam import LangSAM
from PIL import Image
import torchvision.transforms as transforms
import plotly.graph_objects as go
from functools import reduce

"""
.jpg or .png formats is required for images
"""

def seg_img_dir(tool="screw_driver", data_dir='data', mode='train'):
    # train_processed_save_path = f"{data_dir}/{tool}/train_preprocessed.tar"
    train_directory=f"{data_dir}/{tool}/{mode}"

    seg_model = LangSAM()

    seg_prompt = {
        'screw_driver' : "robot with blue screwdriver",
        'hammer' : "hammer held in robot hand",
        'brush' : "silver robot and grey paintbrush",
        'wrench' : "robot claw with red wrench",
        'tape' : "silver robot and roll of red tape"
    }

    #"silver robot claw and a red object with a black and white checkered square piece of paper on it",
    file_names_tr = os.listdir(train_directory)
    file_paths_tr = [os.path.join(train_directory, file) for file in file_names_tr]
    file_paths_tr = [f for f in file_paths_tr if f.endswith('.jpg') or f.endswith('.png')]
    file_paths_tr.sort()
    r_height,r_width=int(480*0.8),int(640*0.8)
    mask_tor_path = f"{train_directory}/mask_tor.pt"

    mask_list=[]
    if f"{tool}_{mode}" in seg_prompt:
        text_prompt = seg_prompt[f"{tool}_{mode}"]
    else:
        text_prompt = seg_prompt[tool]
    for i in range(len(file_paths_tr)):
        cur_path=file_paths_tr[i]
        if 'mask' in cur_path:
            continue
        print(cur_path)
        image_pil = Image.open(cur_path).convert("RGB")
        tor_img=torch.tensor(np.array(image_pil))
        masks, boxes, phrases, logits = seg_model.predict(image_pil, text_prompt)
        # print(masks.shape)
        masks=masks.detach().cpu()
        if 'april' in tool:
            masks = reduce(torch.logical_or, masks)
        else:
            masks = masks[0]
        # print(masks.dtype, anti_mask)

        mask_np = masks.numpy()

        # Convert binary mask to uint8 format (0 or 255)
        mask_np = (mask_np * 255).astype(np.uint8)

        # Create a PIL Image from the numpy array
        mask_image = Image.fromarray(mask_np)

        # Save the image as a PNG file
        mask_image.save(f"{train_directory}/maskof{i}.png")
        
        resize_transform = transforms.Resize((r_height,r_width),
                                            interpolation=transforms.InterpolationMode.NEAREST)
        r_mask = resize_transform((masks.unsqueeze(0).unsqueeze(0)))
        o_mask=r_mask[0,0,:,:]
        mask_list.append(o_mask)
        tor_img[masks==0]=255
        conv_img=Image.fromarray(np.array(tor_img))
        cur_sv_path=train_directory+'/'+f"masked_{i:02}.png"
        conv_img.save(cur_sv_path)
    mask_tor=torch.stack(mask_list)
    torch.save(mask_tor, mask_tor_path)


    print(f"Masked tensor shape: {mask_tor.shape}")
    return mask_tor

if __name__ ==  "__main__":
    
    L = ['screw_driver', 'hammer', 'brush', 'block', 'wrench', 'tape', 
     'ruler', 'scbox', 'wrench_april']
    for tool in L:
        seg_img_dir(tool=tool, data_dir='data', mode='test')
        seg_img_dir(tool=tool, data_dir='data', mode='train')
        