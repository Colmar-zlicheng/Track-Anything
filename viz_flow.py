import os
import cv2
import shutil
import argparse
import numpy as np
from tools.etqdm import etqdm
from tools.img_to_mp4 import image_to_video


def colormap(rgb=True):
    color_list = np.array([
        0.000, 0.000, 0.000, 1.000, 1.000, 1.000, 1.000, 0.498, 0.313, 0.392, 0.581, 0.929, 0.000, 0.447, 0.741, 0.850,
        0.325, 0.098, 0.929, 0.694, 0.125, 0.494, 0.184, 0.556, 0.466, 0.674, 0.188, 0.301, 0.745, 0.933, 0.635, 0.078,
        0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600, 1.000, 0.000, 0.000, 1.000, 0.500, 0.000, 0.749, 0.749, 0.000,
        0.000, 1.000, 0.000, 0.000, 0.000, 1.000, 0.667, 0.000, 1.000, 0.333, 0.333, 0.000, 0.333, 0.667, 0.000, 0.333,
        1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000, 0.667, 1.000, 0.000, 1.000, 0.333, 0.000, 1.000, 0.667,
        0.000, 1.000, 1.000, 0.000, 0.000, 0.333, 0.500, 0.000, 0.667, 0.500, 0.000, 1.000, 0.500, 0.333, 0.000, 0.500,
        0.333, 0.333, 0.500, 0.333, 0.667, 0.500, 0.333, 1.000, 0.500, 0.667, 0.000, 0.500, 0.667, 0.333, 0.500, 0.667,
        0.667, 0.500, 0.667, 1.000, 0.500, 1.000, 0.000, 0.500, 1.000, 0.333, 0.500, 1.000, 0.667, 0.500, 1.000, 1.000,
        0.500, 0.000, 0.333, 1.000, 0.000, 0.667, 1.000, 0.000, 1.000, 1.000, 0.333, 0.000, 1.000, 0.333, 0.333, 1.000,
        0.333, 0.667, 1.000, 0.333, 1.000, 1.000, 0.667, 0.000, 1.000, 0.667, 0.333, 1.000, 0.667, 0.667, 1.000, 0.667,
        1.000, 1.000, 1.000, 0.000, 1.000, 1.000, 0.333, 1.000, 1.000, 0.667, 1.000, 0.167, 0.000, 0.000, 0.333, 0.000,
        0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000,
        0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000,
        0.000, 0.167, 0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000,
        1.000, 0.143, 0.143, 0.143, 0.286, 0.286, 0.286, 0.429, 0.429, 0.429, 0.571, 0.571, 0.571, 0.714, 0.714, 0.714,
        0.857, 0.857, 0.857
    ]).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list


color_list = colormap()
color_list = color_list.astype('uint8').tolist()


def vis_add_mask(image, mask, color, alpha):
    color = np.array(color_list[color])
    mask = mask > 0.5
    image[mask] = image[mask] * (1 - alpha) + color * alpha
    return image.astype('uint8')


def add_mask(img, mask, contour_width=3, mask_color=5, mask_alpha=0.7, contour_color=1, draw_contour=True):
    mask = np.clip(mask, 0, 1)
    painted_image = vis_add_mask(img.copy(), mask.copy(), mask_color, mask_alpha)

    if draw_contour:
        contour_radius = (contour_width - 1) // 2

        dist_transform_fore = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        dist_transform_back = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)
        dist_map = dist_transform_fore - dist_transform_back

        contour_radius += 2
        contour_mask = np.abs(np.clip(dist_map, -contour_radius, contour_radius))
        contour_mask = contour_mask / np.max(contour_mask)
        contour_mask[contour_mask > 0.5] = 1.

        painted_image = vis_add_mask(painted_image.copy(), 1 - contour_mask, contour_color, 1)

    return painted_image


def viz_save_flow(args):
    masks_root = os.path.join(args.data_path, 'masks')
    masks_dir = sorted(os.listdir(masks_root))

    for tmp_dir in masks_dir:
        if tmp_dir == 'mask_bkg' or tmp_dir.endswith('.mp4'):
            masks_dir.remove(tmp_dir)

    obj_num = len(masks_dir)

    masks = []
    for i in etqdm(range(obj_num)):
        masks_path_i = sorted(os.listdir(os.path.join(masks_root, masks_dir[i])))
        for j in range(len(masks_path_i)):
            masks_path_i[j] = (cv2.imread(os.path.join(masks_root, masks_dir[i], masks_path_i[j]), 0) /
                               255).astype('uint8')
        masks.append(masks_path_i)

    img_num = len(masks[0])

    images = []
    images_path = sorted(os.listdir(os.path.join(args.data_path, 'images')))
    for i in etqdm(range(img_num)):
        images.append(cv2.imread(os.path.join(args.data_path, 'images', images_path[i])))

    for i in etqdm(range(img_num)):
        img = images[i]

        for n in range(obj_num):
            mask_alpha = args.mask_alpha
            alphas = []

            if i < args.cache_frame:
                for j in range(0, i + 1):
                    mask_alpha = mask_alpha * (1 - 1 / args.cache_frame)
                    alphas.append(mask_alpha)
                for j in range(0, i + 1):
                    alpha = alphas.pop()
                    draw_contour = True if j == i else False
                    img = add_mask(img, masks[n][j], mask_color=n + 2, mask_alpha=alpha, draw_contour=draw_contour)

            else:
                for j in range(i - args.cache_frame, i + 1):
                    mask_alpha = mask_alpha * (1 - 1 / args.cache_frame)
                    alphas.append(mask_alpha)
                for j in range(i - args.cache_frame, i + 1):
                    alpha = alphas.pop()
                    draw_contour = True if j == i else False
                    img = add_mask(img, masks[n][j], mask_color=n + 2, mask_alpha=mask_alpha, draw_contour=draw_contour)

        cv2.imwrite(os.path.join(args.save_path, f'{i:04}.jpg'), img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--data_path', type=str, required=True)
    parser.add_argument('-s', '--save_path', type=str, default=None)
    parser.add_argument('-f', '--cache_frame', type=int, default=10)
    parser.add_argument('--mask_alpha', type=float, default=0.7)
    arg = parser.parse_args()

    if arg.save_path is None:
        arg.save_path = os.path.join(arg.data_path, 'flow')

    save_path = os.path.join(arg.save_path)
    if os.path.exists(save_path):
        if input(f'Directory {save_path} already exists. Override? [Y/N]: ').lower() == 'y':
            shutil.rmtree(save_path)
        else:
            exit()
    os.mkdir(save_path)

    viz_save_flow(args=arg)

    image_to_video(file=arg.save_path, output=os.path.join(arg.data_path, 'flow.mp4'), fps=10)
