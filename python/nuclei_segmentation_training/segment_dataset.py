import sys
from pathlib import Path

from tifffile import imread

import numpy as np
import torch

from scipy.ndimage import zoom


sys.path.insert(0, "../utils")
from norm_percentile_nocrop import norm_percentile_nocrop
from utils.predict_by_parts import predict_by_parts
from utils.organized import split_nuclei, balloon

src_path = Path(
    "/Users/pszyc/Library/CloudStorage/GoogleDrive-przemek.7678@gmail.com/My Drive/Studia/Ogniska/NHDF/"
)
model_path = "segmentation_model.pt"
img_folders = sorted(src_path.glob("*"))


resized_img_size = [505, 681, 48]  # image is resized to this size
normalization_percentile = 0.0001  # image is normalized into this percentile range
mask_erosion = [14, 14, 5]  # amount of mask erosion (elipsoid)

minimal_nuclei_size = 10000
h = 2
min_dist = 35

crop_size = [96, 96]

device = torch.device("cpu")

X, Y, Z = np.meshgrid(
    np.linspace(-1, 1, mask_erosion[0]),
    np.linspace(-1, 1, mask_erosion[1]),
    np.linspace(-1, 1, mask_erosion[2]),
)
sphere = np.sqrt(X**2 + Y**2 + Z**2) < 1


model = torch.load(model_path, map_location=torch.device("cpu"))


def predict_nn(data_path):
    img_path = data_path / "data_53BP1.tif"
    img_filename = str(img_path)
    img = []
    img.append(imread(img_filename))  # red
    img.append(imread(img_filename.replace("53BP1", "gH2AX")))  # green
    img.append(imread(img_filename.replace("53BP1", "DAPI")))  # DAPI
    img = np.stack(img, axis=3)

    original_img_size = img.shape[:3]

    factor = np.array(resized_img_size) / np.array(original_img_size)

    tmp_size = resized_img_size.copy()
    tmp_size.append(img.shape[3])
    img_resized = np.zeros(tmp_size, dtype=np.float32)
    for channel in range(img.shape[3]):

        data_one_channel = img[..., channel]

        data_one_channel = zoom(data_one_channel, factor, order=1)
        data_one_channel = norm_percentile_nocrop(
            data_one_channel, normalization_percentile
        )

        img_resized[..., channel] = data_one_channel

    img = img_resized

    img = img.astype(np.float32)
    img = np.transpose(img, (3, 0, 1, 2)).copy()
    img = torch.from_numpy(img)
    img = img.to(device)

    mask_predicted = predict_by_parts(model, img, crop_size=crop_size)
    mask_predicted = mask_predicted.detach().cpu().numpy()[0, :, :, :]
    mask_predicted -= mask_predicted.min()
    mask_predicted /= mask_predicted.max()

    mask_split = split_nuclei(
        mask_predicted > 0.5, minimal_nuclei_size, h, sphere, min_dist
    )

    mask_label_dilated = balloon(mask_split, sphere)

    factor = np.array(original_img_size) / np.array(mask_label_dilated.shape)
    mask_final = zoom(mask_label_dilated, factor, order=0)
    return mask_final


data_files = list(src_path.rglob("00*"))

data_files = [path for path in data_files if not (path / "nuclei_mask.npy").exists()]
print(len(data_files))


from tqdm import tqdm

for data_path in tqdm(data_files):
    output_path = data_path / "nuclei_mask.npy"
    if output_path.exists():
        continue
    mask_final = predict_nn(data_path)
    np.save(output_path, mask_final)
