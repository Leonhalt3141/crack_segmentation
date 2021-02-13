##
import os
import numpy as np
from typing import Tuple, List
from glob import glob
from tqdm import tqdm
import cv2 as cv
from create_one_image import run as create_one_image_run


def random_crop(img: np.ndarray, mask: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    assert img.shape[0] == mask.shape[0]
    assert img.shape[1] == mask.shape[1]

    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[0] - height)
    slice_img = img[y:y + height, x:x + width]
    slice_mask = mask[y:y + height, x:x + width]
    return slice_img, slice_mask


def create_mask(img: np.ndarray) -> np.ndarray:
    # mask = np.zeros((height, width), dtype=np.float32)
    # height, width, band = img.shape
    mask = np.sum(img, axis=2) / 3
    mask[mask == 255] = 0
    mask[mask != 0] = 1
    return mask


def export_image_array(img: np.ndarray, mask: np.ndarray, image_name: str, width: int = 448, height: int = 448,
                       export_path: str = 'data/crack_image/BridgeCrack/inference'):
    exportPathExists = os.path.exists(export_path)
    if not exportPathExists:
        os.makedirs(export_path, exist_ok=True)

    crop, crop_mask = random_crop(img, mask, width, height)
    image_path = os.path.join(export_path, image_name)
    n = crop_mask[crop_mask != 0].shape[0]

    if n <= width * height * 0.9:
        return 0
    else:
        cv.imwrite(image_path, cv.cvtColor(crop, cv.COLOR_RGBA2BGR))
        return 1


def export_image_file(image_path: str, export_root_path: str = 'data/crack_image/BridgeCrack/inference',
                      width: int = 448, height: int = 448):
    N = 5000
    checkSize = 500

    file_name = os.path.basename(image_path)
    img = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)
    mask = create_mask(img)

    index = 1
    while index <= N:
        image_path = file_name.split('.')[0] + '_%05d.jpg' % index
        result = export_image_array(img, mask, image_path, width, height, export_root_path)

        if result == 1:
            index += 1
            indexCheckSize = index % checkSize == 0
            if indexCheckSize:
                print('%05d/%05d' % (index, N))


def get_all_files(root_path: str = 'data/BridgeCrack/') -> List[str]:
    files: List[str] = glob(os.path.join(root_path, '*.tif'))
    files += glob(os.path.join(root_path, '*/*.tif'))
    return files


def process_all_files():
    image_files = get_all_files()
    image_count = len(image_files)
    for ix, image_file in enumerate(image_files):
        print('Processing %s (%d/%d)' % (image_file, ix, image_count))
        export_image_file(image_file)
##

process_all_files()
##

image_file = 'bridge_end.tif'
image_path = os.path.join('/home/ken/workspace/crack_segmentation/data/BridgeCrack/', image_file)
down_index = 100
output_path = os.path.join('/home/ken/workspace/crack_segmentation/data/crack_image/inference_result/', image_file)
create_one_image_run(image_path=image_path, output_path=output_path)
##

image_file = 'bridge_start.tif'
image_path = os.path.join('/home/ken/workspace/crack_segmentation/data/BridgeCrack/', image_file)
down_index = 100
output_path = os.path.join('/home/ken/workspace/crack_segmentation/data/crack_image/inference_result/', image_file)
create_one_image_run(image_path=image_path, output_path=output_path)
##

image_file = 'bridgeL.tif'
image_path = os.path.join('/home/ken/workspace/crack_segmentation/data/BridgeCrack/', image_file)
down_index = 100
output_path = os.path.join('/home/ken/workspace/crack_segmentation/data/crack_image/inference_result/', image_file)
create_one_image_run(image_path=image_path, output_path=output_path)
##

image_file = 'bridgeR2.tif'
image_path = os.path.join('/home/ken/workspace/crack_segmentation/data/BridgeCrack/', image_file)
down_index = 100
output_path = os.path.join('/home/ken/workspace/crack_segmentation/data/crack_image/inference_result/', image_file)
create_one_image_run(image_path=image_path, output_path=output_path)
##

image_file = 'SideWall/Base_Ortho.tif'
image_path = os.path.join('/home/ken/workspace/crack_segmentation/data/BridgeCrack/', image_file)
down_index = 100
output_path = os.path.join('/home/ken/workspace/crack_segmentation/data/crack_image/inference_result/', os.path.basename(image_file))
create_one_image_run(image_path=image_path, output_path=output_path)
##

image_file = 'SideWall/SideWall_L_Ortho.tif'
image_path = os.path.join('/home/ken/workspace/crack_segmentation/data/BridgeCrack/', image_file)
down_index = 100
output_path = os.path.join('/home/ken/workspace/crack_segmentation/data/crack_image/inference_result/', os.path.basename(image_file))
create_one_image_run(image_path=image_path, output_path=output_path)
##

image_file = 'SideWall/SideWall_R_Ortho.tif'
image_path = os.path.join('/home/ken/workspace/crack_segmentation/data/BridgeCrack/', image_file)
down_index = 100
output_path = os.path.join('/home/ken/workspace/crack_segmentation/data/crack_image/inference_result/', os.path.basename(image_file))
create_one_image_run(image_path=image_path, output_path=output_path)
##

image_file = 'SideWall/sub/DownBase.tif'
image_path = os.path.join('/home/ken/workspace/crack_segmentation/data/BridgeCrack/', image_file)
down_index = 100
output_path = os.path.join('/home/ken/workspace/crack_segmentation/data/crack_image/inference_result/', os.path.basename(image_file))
create_one_image_run(image_path=image_path, output_path=output_path)
##

image_file = 'SideWall/sub/DownSideL.tif'
image_path = os.path.join('/home/ken/workspace/crack_segmentation/data/BridgeCrack/', image_file)
down_index = 100
output_path = os.path.join('/home/ken/workspace/crack_segmentation/data/crack_image/inference_result/', os.path.basename(image_file))
create_one_image_run(image_path=image_path, output_path=output_path)
##
image_file = 'SideWall/sub/Upside.tif'
image_path = os.path.join('/home/ken/workspace/crack_segmentation/data/BridgeCrack/', image_file)
down_index = 100
output_path = os.path.join('/home/ken/workspace/crack_segmentation/data/crack_image/inference_result/', os.path.basename(image_file))
create_one_image_run(image_path=image_path, output_path=output_path)

##

image_file = 'Matsuyama_AreaG.png'
image_path = os.path.join('/home/ken/workspace/crack_segmentation/crack_image_old', image_file)
output_path = os.path.join('/home/ken/workspace/crack_segmentation/data/crack_image/inference_result/', os.path.basename(image_file))
create_one_image_run(image_path=image_path, output_path=output_path)
##

