
import cv2 as cv
from skimage import measure
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
from utils import load_unet_resnet_101
from prepare_image import split_image
import numpy as np


model = load_unet_resnet_101('models/model_best.pt')
width = 448
height = 448
channel_means = [0.485, 0.456, 0.406]
channel_stds = [0.229, 0.224, 0.225]
threshold = 0.2
train_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(channel_means, channel_stds)])


def evaluate_img(model_, img):
    img_1 = cv.resize(img, (width, height), cv.INTER_AREA)
    X = train_tfms(Image.fromarray(img_1))
    X = Variable(X.unsqueeze(0)).cuda()  # [N, 1, H, W]

    mask = model_(X)

    mask = F.sigmoid(mask[0, 0]).data.cpu().numpy()
    mask = cv.resize(mask, (width, height), cv.INTER_AREA)
    return mask


def evaluate_img_patch(model, img):
    img_height, img_width, img_channels = img.shape

    if img_width < width or img_height < height:
        return evaluate_img(model, img)

    stride_ratio = 0.1
    stride = int(width * stride_ratio)

    normalization_map = np.zeros((img_height, img_width), dtype=np.int16)

    patches = []
    patch_locs = []
    for y in range(0, img_height - height + 1, stride):
        for x in range(0, img_width - width + 1, stride):
            segment = img[y:y + height, x:x + width]
            normalization_map[y:y + height, x:x + width] += 1
            patches.append(segment)
            patch_locs.append((x, y))

    patches = np.array(patches)
    if len(patch_locs) <= 0:
        return None

    preds = []
    for i, patch in enumerate(patches):
        patch_n = train_tfms(Image.fromarray(patch))
        X = Variable(patch_n.unsqueeze(0)).cuda()  # [N, 1, H, W]
        masks_pred = model(X)
        mask = F.sigmoid(masks_pred[0, 0]).data.cpu().numpy()
        preds.append(mask)

    probability_map = np.zeros((img_height, img_width), dtype=float)
    for i, response in enumerate(preds):
        coords = patch_locs[i]
        probability_map[coords[1]:coords[1] + height, coords[0]:coords[0] + width] += response

    return probability_map


def run(image_path, down_index, output_path):
    img = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)

    mask = np.sum(img, axis=2) / 3
    mask[mask == 255] = 0
    mask[mask != 0] = 1

    out_masks = split_image(mask)
    out_imgs = split_image(img)

    results = []
    for i, (out_img, out_mask) in enumerate(zip(out_imgs, out_masks)):
        n = out_mask[out_mask == 1].shape[0]
        if n <= width * height * 0.9:
            results.append(np.zeros((height, width),dtype=np.float32))
        else:
            prob_map_full = evaluate_img(model, out_img)
            prob_map_full[prob_map_full < 0.2] = 0
            results.append(prob_map_full)

    out_imgs = np.array(results)
    h, w = img.shape[:2]
    num_vsplits, num_hsplits = np.floor_divide([h, w], [width, height])
    split_imgs = out_imgs.reshape((num_vsplits, num_hsplits, width, height, 1))

    merged_img = np.vstack([np.hstack(h_imgs) for h_imgs in split_imgs])
    merged_img = merged_img.reshape(merged_img.shape[0], merged_img.shape[1])

    crack_img = np.zeros(img.shape, dtype=np.uint8)
    crack_img[:merged_img.shape[0], :merged_img.shape[1], 0] = np.uint8(merged_img * 255)

    label = crack_img.copy()[:, :, 0]
    label[label > 0] = 1
    label = measure.label(label, background=0, neighbors=8, connectivity=2)

    vals = np.linspace(0, 1, label.max())
    np.random.shuffle(vals)
    color_map = plt.cm.colors.ListedColormap(plt.cm.jet(vals))
    norm = plt.Normalize(vmin=0, vmax=float(label[label != 0].max()))
    image = color_map(norm(label))
    image = image[:, :, :3]
    image[label == 0, :] = 0
    image[:, :, :3] = image[:, :, :3] * 255
    image = np.uint8(image)
    image[down_index:, :, :] = 0

    dummy = cv.addWeighted(image, 0.7, img, 1, 2.2)

    cv.imwrite(output_path, cv.cvtColor(dummy, cv.COLOR_BGR2RGB))
