from src.image_handling import open_image, save_image, batch_process
from src.const import original_dir_name, subsampled_dir_name


def subsample(relative_pathfile):
    tensor = open_image(original_dir_name, relative_pathfile)
    subsampled_tensor = tensor[:, ::2, ::2]
    save_image(relative_pathfile, subsampled_dir_name, subsampled_tensor)


if __name__ == '__main__':
    batch_process(subsample, original_dir_name)
