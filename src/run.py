from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF

from src.const import root_dir, dl_dir_name
from src.interpolation import bicubic_interpolate_tensor
from src.srcnn import SRCNN


def upscale(dir_name, relative_pathfile, scale_width=2, scale_height=2):
    img = Image.open(Path(root_dir, dir_name, relative_pathfile)).convert('YCbCr')
    tensor = TF.to_tensor(img)
    tensor = bicubic_interpolate_tensor(tensor, scale_width, scale_height)

    tensor = tensor.unsqueeze(0)
    y = tensor[:, :1]
    cb = tensor[:, 1] * 255.0
    cr = tensor[:, 2] * 255.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SRCNN().to(device)
    model.load_state_dict(torch.load('../output/model.pth'))
    input_data = y.to(device)

    output_data = model(input_data)
    output_data = output_data.cpu()
    output_y = output_data.detach().numpy()
    output_y *= 255.0
    output_y = output_y.clip(0, 255)

    output_y = Image.fromarray(np.uint8(output_y[0][0]), mode='L')
    cb = Image.fromarray(np.uint8(cb[0]), mode='L')
    cr = Image.fromarray(np.uint8(cr[0]), mode='L')

    # merge the output of our network with the upscaled Cb and Cr from before
    out_img = Image.merge('YCbCr', [output_y, cb, cr]).convert('RGB')

    save_path = Path(root_dir, dl_dir_name, relative_pathfile)
    save_path_parent = save_path.parent

    # create folder if needed
    if not Path.exists(save_path_parent):
        Path.mkdir(save_path_parent, parents=True)

    out_img.save(f"../images/{dl_dir_name}/{relative_pathfile}")


if __name__ == '__main__':
    upscale('subsampled', 'airplane/airplane00.tif')
    
