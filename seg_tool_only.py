from lang_sam import LangSAM
from PIL import Image
import numpy as np
import re, os, torch

from functools import reduce
import matplotlib.pyplot as plt
from torchvision import transforms

seg_model = LangSAM()

def is_original_images(file):
    file = file.split('/')[-1]
    pattern = re.compile(r'\d{2}\.(jpg|png)')
    return pattern.fullmatch(file)

components = {
    'screw_driver': ['a light gray color screwdriver blade', "medium blue color handle with a sticker on it"],
}

def mask_resize(masks):
    mask_np = masks.numpy()
    r_height,r_width=int(480*0.8),int(640*0.8)
    # Convert binary mask to uint8 format (0 or 255)
    mask_np = (mask_np * 255).astype(np.uint8)
    
    resize_transform = transforms.Resize((r_height,r_width),
                                        interpolation=transforms.InterpolationMode.NEAREST)
    r_mask = resize_transform((masks.unsqueeze(0).unsqueeze(0)))
    o_mask=r_mask[0,0,:,:]
    return o_mask


def seg_tool_only(tool, data_dir='data', tool_tip_only=False):
    data_dir_train = f'{data_dir}/{tool}/train'

    filenames = os.listdir(data_dir_train)
    filenames.sort()
    org_images = []
    # Copy images to the temporary folder
    for filename in filenames:
        if (filename.endswith(('.jpg', '.jpeg', '.png', '.gif'))):
            original_path = os.path.join(data_dir_train, filename)
            if os.path.isfile(original_path) and is_original_images(original_path):
                org_images.append(original_path)
                #print(original_path)

    mask_all = []
    for i in range(9):
        image_pil = Image.open(org_images[i]).convert("RGB")
        tor_img=torch.tensor(np.array(image_pil))
        
        masks_tool = []
        plt.imshow(tor_img)
        if tool_tip_only:
            components_used = components[tool][:1]
        else:
            components_used = components[tool]
        for c in components_used:
            masks_tool_c, _, _, _ = seg_model.predict(image_pil, c) 
            masks_tool_c = masks_tool_c.detach().cpu()
            masks_tool_c = masks_tool_c[0]

            masks_tool.append(masks_tool_c)
        masks_tool = reduce(torch.logical_or, masks_tool)
        mask_all.append(mask_resize(masks_tool))
        plt.imshow(masks_tool.numpy(), cmap='plasma', alpha=0.5)
        plt.axis('off')  # Hide axes
        plt.show()
    if tool_tip_only:
        torch.save(mask_all, f'{data_dir}/{tool}/{tool}_tip_masks.pt')
    else:
        torch.save(mask_all, f'{data_dir}/{tool}/{tool}_masks.pt')
    return mask_all

if __name__ == "__main__":
    tool = 'screw_driver'
    seg_tool_only(tool=tool, data_dir='data', tool_tip_only=False)