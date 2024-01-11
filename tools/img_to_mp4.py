import os
import cv2
from tools.etqdm import etqdm


def image_to_video(file, output, fps=10):
    num = sorted(os.listdir(file))
    tmp_img = cv2.imread(os.path.join(file, num[0]))
    H, W = tmp_img.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videowriter = cv2.VideoWriter(output, fourcc, fps, (W, H))
    bar = etqdm(num)
    for n in bar:
        path = os.path.join(file, n)
        frame = cv2.imread(path)
        videowriter.write(frame)
        bar.set_description(path)

    videowriter.release()
