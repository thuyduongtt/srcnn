from pathlib import Path
from PIL import Image
from torchvision.transforms import functional as TF

from src.const import root_dir, original_dir_name


def open_image(dir_name, relative_pathfile):
    img = Image.open(Path(root_dir, dir_name, relative_pathfile))
    tensor = TF.to_tensor(img)
    tensor.unsqueeze(0)
    return tensor


def save_image(relative_pathfile, save_dir, tensor):
    # prevent overwriting
    assert save_dir != original_dir_name, 'Do not overwrite the original data!'

    save_path = Path(root_dir, save_dir, relative_pathfile)
    save_path_parent = save_path.parent

    # create folder if needed
    if not Path.exists(save_path_parent):
        Path.mkdir(save_path_parent, parents=True)

    img = TF.to_pil_image(tensor)
    img.save(save_path)


def batch_process(process_func, dir_name, sub_dir_name=None):
    if sub_dir_name is not None:
        dir_path = Path(root_dir, dir_name, sub_dir_name)
        if not Path.exists(dir_path):
            Path.mkdir(dir_path, parents=True)

        for file in dir_path.iterdir():
            process_func(str(Path(sub_dir_name, file.name)))

    else:
        dir_path = Path(root_dir, dir_name)
        for folder in dir_path.iterdir():
            print('====== ' + str(folder.name))
            batch_process(process_func, dir_name, folder.name)


if __name__ == '__main__':
    batch_process(print, original_dir_name)
