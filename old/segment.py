#!/bin/python3
import cv2
import numpy as np
from sklearn.cluster import KMeans
from functools import cmp_to_key
from PIL import Image
from skimage.filters import difference_of_gaussians

import torch
import torch.nn.functional as F

from multiprocessing import Pool

class PlantProcessor():

    def __init__(self, filename):
        """Init the plant processor"""
        self.filename = filename
        self.color_set = set()
        self.local_green = np.array([0, 255, 0])

    def _process_over_frames(self, fctor, frame_id=None):
        """Apply fctor to each frame"""
        cap = cv2.VideoCapture(self.filename)
        if not cap.isOpened():
            print(f"Could not open file: {self.filename}")
            return list()

        print(f"Reading file: {self.filename}")
        frame_count = 0
        ret, frame = cap.read()
        while ret:
            frame_count += 1
            ret, frame = cap.read()
            if frame_id is None:
                out = fctor(frame)
            elif frame_id is not None and frame_count == frame_id:
                out = fctor(frame)
                return out


        print(f"{frame_count} frames.")
        cap.release()

    def preprocess_video(self):
        """sample frames and get distribution of color."""
        cap = cv2.VideoCapture(self.filename)
        if not cap.isOpened():
            raise Exception(f"Could not open file: {self.filename}")

        def prep_color_points(frame):
            """Prep the color points into numpy for clustering"""

        print("Sampling frames...")
        frame_count = 0
        ret, frame = cap.read()
        sampled_frames = list()
        while ret:
            ret, frame = cap.read()
            if frame_count in (5, 10, 12, 18):
                sampled_frames.append(frame)
            
            frame_count+=1

        print(f"Sampled {len(sampled_frames)} frames.")
        cap.release()

        color_points = np.concatenate([
            x.reshape((1080*1920, 3)) for x in sampled_frames
        ])

        colors = list(set([tuple(x) for x in  color_points.tolist()]))

        print("Get greenest pixels")
        def green_cmp_fctor(a, b):
            if (a[1]-(a[0]+a[2])/2.0) < (b[1]-(b[0]+b[2])/2.0):
                return 1
            if (a[1]-(a[0]+a[2])/2.0) > (b[1]-(b[0]+b[2])/2.0):
                return -1
            return 0

        colors = sorted(colors, key=cmp_to_key(green_cmp_fctor))
        greens = np.array(colors[:int(len(colors)*0.05)])
        green = greens.mean(axis=0)

        self.local_green = green
        #self.frame_size = (np.shape(frame)[0], np.shape(frame)[1])
        self.fps = 60

def process_video(filename, local_green):
    """Process each frame and identify the plants"""

    BATCH = 30
    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{total_frames} to process.")
    if not cap.isOpened():
        print(f"Could not open file: {filename}")
        return list()

    print(f"Reading file: {filename}")
    frame_count = 0
    ret, frame = cap.read()

    height, width = np.shape(frame)[0], np.shape(frame)[1]
    print(height, width)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

    frame_buffer = list()
    while ret:
        frame_count += 1
        ret, frame = cap.read()
        frame_buffer.append(frame)
        if len(frame_buffer) >= BATCH:
            print(f"{(frame_count/total_frames)*100:.2f}%")
            frame_buffer = np.stack(frame_buffer, axis=0)
            out_frame = mark_bb(frame_buffer, local_green)

            print(np.shape(out_frame))
            for i in range(len(out_frame)):
                writer.write(out_frame[i])
            frame_buffer = list()

    print(f"{frame_count} frames.")
    cap.release()
    writer.release()

def mark_bb(frames, local_green):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print("Creating mask")
    B, x, y, _ = np.shape(frames)
    threshold = 20
    mask = np.linalg.norm(local_green - frames, axis=3) < threshold

    kernel = np.ones((8, 8), np.uint8)
    masks = list()
    for i in range(B):
        masks.append(cv2.morphologyEx(
            mask[i].astype(np.uint8) * 255,
            cv2.MORPH_OPEN,
            kernel
        ))
    masks = np.stack(masks, axis=0)

    # Repeatedly expand the filter if the adjacent pixels are close enough
    # to the color
    threshold2 = 30
    changed = True
    max_depth = 30

    img = torch.from_numpy(frames).float().permute(0, 3, 1, 2)
    mask = (torch.from_numpy(masks).float() > 0).float()
    mask = mask.unsqueeze(1)
    B, H, W = img.shape[0], img.shape[2], img.shape[3]
    img_p = F.unfold(img,  kernel_size=3, padding=1)
    img_p = img_p.view(B, 3, 9, H, W)

    print("Masking plants")
    for i in range(max_depth):
        mask_p = F.unfold(mask, kernel_size=3, padding=1)
        mask_p = mask_p.view(B, 1, 9, H, W)

        center_rgb = img_p[:, :, 4]
        neighbor_rgb = img_p[:, :, [0,1,2,3,5,6,7,8]]
        neighbor_mask = mask_p[:, :, [0,1,2,3,5,6,7,8]]

        diff = torch.abs(center_rgb.unsqueeze(2) - neighbor_rgb)
        diff_rgb = diff.mean(dim=1)
        masked_diff = diff_rgb*neighbor_mask[:, 0]

        alive_count = neighbor_mask[:, 0].sum(dim=1)
        has_alive = alive_count > 0

        score = torch.abs(masked_diff.sum(dim=1) / (alive_count + 1e-6))
        flip = (score < threshold2) & has_alive
        flip_allowed = flip & (mask[:,0] == 0)

        new_mask = mask.clone()
        new_mask[:, 0] = torch.where(flip_allowed, 1.0, mask[:, 0])
        mask = new_mask

    masks = mask.cpu().numpy().squeeze(1)

    #while changed and max_depth != 0:
    #    max_depth -= 1
    #    changed = False
    #    for i in range(1, x-1):
    #        for j in range(1, y-1):
    #            if filtered[i][j] > 0:
    #                continue
    #            mask = filtered[i-1:i+2, j-1:j+2] > 0
    #            if mask.sum() == 0:
    #                continue
    #            patch = frame[i-1:i+2, j-1:j+2, :].astype(np.int8)
    #            l1d = np.abs((patch[1][1] - patch[mask]).sum()/(mask.sum()*3))
    #            rgb = patch[1][1]
    #            greenish = rgb[1] > max(rgb[0], rgb[2])
    #            if l1d < threshold2 and greenish:
    #                changed = True
    #                filtered[i][j] = 1

    
    print("Drawing BBs")
    with Pool(20) as p:
        out = np.stack(p.map(frame_draw_bb, [(frames[i], masks[i]) for i in range(B)]), axis=0)

    return out

# must be outside of class to multiprocess
def frame_draw_bb(args):
    frame, mask = args
    #global FRAME_BUFFER, MASK_FRAMES
    kernel = np.ones((16, 16), np.uint8)
    ifilter = cv2.morphologyEx(
        mask.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        kernel
    )

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        ifilter,
        connectivity=8
    )

    #frame[ifilter > 0] = [255, 0, 0]
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    return frame


def main():
    p_proc = PlantProcessor("videos/20260121_comp_2221_2114.mp4")
    p_proc.preprocess_video()
    process_video(p_proc.filename, p_proc.local_green)


if __name__ == "__main__":
    main()
