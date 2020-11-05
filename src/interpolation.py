from src.image_handling import open_image, save_image, batch_process
from src.const import subsampled_dir_name, interpolated_dir_name
import torch.nn.functional as F


def bicubic_interpolate_image(relative_pathfile, scale_width=2, scale_height=2):
    tensor = open_image(subsampled_dir_name, relative_pathfile)
    itpl_tensor = bicubic_interpolate_tensor(tensor, scale_width, scale_height)
    save_image(relative_pathfile, interpolated_dir_name, itpl_tensor)


def bicubic_interpolate_tensor(tensor, scale_width=2, scale_height=2):
    tensor = tensor.unsqueeze(0)  # add batch dimension
    itpl_tensor = F.interpolate(tensor, scale_factor=(scale_height, scale_width), mode='bicubic')
    itpl_tensor = itpl_tensor.squeeze(0)
    itpl_tensor = itpl_tensor.clamp(0, 1)
    return itpl_tensor


if __name__ == '__main__':
    batch_process(bicubic_interpolate_image, subsampled_dir_name)
