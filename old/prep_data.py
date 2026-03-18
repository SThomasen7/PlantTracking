#!/home/stasen/Desktop/plantas/env/bin/python
import cv2
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from functools import cmp_to_key
from PIL import Image
from skimage.filters import difference_of_gaussians
from datetime import datetime

import torch
import torch.nn.functional as F

from multiprocessing import Pool


MISSING_FRAME_LIMIT = 5

class PlantProcessor():

    def __init__(self, filename, batch_size=20):
        """Init the plant processor"""
        self.filename = filename
        self.batch_size = batch_size
        self.fps = -1
        self.total_frames = -1 
        self.width = 0
        self.height = 0
        self.writer = None
        os.makedirs("dataset/images/all", exist_ok=True)
        os.makedirs("dataset/labels/all", exist_ok=True)
        os.makedirs("dataset/images_annotated/", exist_ok=True)

        with open("dataset/data.yaml", 'w') as fptr:
            fptr.write("names:\n  0: planta")

    def process(self):

        # open the video and get metadata
        print(f"Reading {self.filename}")
        cap = cv2.VideoCapture(self.filename)
        if not cap.isOpened():
            raise Exception(f"Could not open file: {self.filename}")
        self.fps = cap.get(cv2.CAP_PROP_FPS)#/3
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._open_writer()

        # read through the frames and process each batch
        print("Processing...")
        ret, frame = cap.read()
        frames = list()
        batch_id = 0
        frame_id = 0
        while ret:
            ret, frame = cap.read()
            frames.append(frame)
            if len(frames) == self.batch_size:
                print(batch_id)
                self._process_batch(np.array(frames), frame_id+1-self.batch_size)
                frames = list()
                batch_id += 1
                #if batch_id > 6:
                    #break
            frame_id += 1

        self._close_writer()
        cap.release()

    def _process_batch(self, frames, start_idx):
        green_data, reconstructed = self._get_green_component(frames)
        threshold = self._get_threshold(green_data)
        mask = np.reshape(green_data, (len(frames), self.height, self.width))
        reconstructed = np.reshape(reconstructed, (len(frames), self.height, self.width, 3))

        kernel = np.ones((8,8),np.uint8)
        for i in range(len(frames)):
            mask[i] = cv2.morphologyEx(mask[i],cv2.MORPH_CLOSE,kernel, iterations=3)
            mask[i] = cv2.morphologyEx(mask[i],cv2.MORPH_OPEN,kernel, iterations=2)
            #mask[i] = cv2.medianBlur(mask[i],2)


        #masked_frames[mask > threshold, :] = [255,0,0]
        MIN_AREA_RATIO = 0.001
        min_area = MIN_AREA_RATIO*self.height*self.width

        for i in range(len(frames)):
            _, tmask = cv2.threshold(mask[i].astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)
            contours,_ = cv2.findContours(tmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = list()
            for cnt in contours:
                #print(cv2.contourArea(cnt))
                if cv2.contourArea(cnt) < min_area:
                    continue
                M = cv2.moments(cnt)
                center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
                if center[0] > 1300:
                    continue
                valid_contours.append(cnt)

            if i % 10 == 0:
                self._write_frame_as_image(frames[i], f"dataset/images/all/", f"{i+start_idx}")
                annotated = self._draw_on_frame(frames[i], valid_contours)
                self._write_annotation_notes(i+start_idx, valid_contours)
                self._write_frame_as_image(frames[i], f"dataset/images_annotated/", f"{i+start_idx}")


    def _get_green_component(self, frames):
        # first we'll get the PCA and find the greenest component
        color_points = np.concatenate([
            x.reshape((self.height*self.width, 3)) for x in frames
        ]).astype(np.float64)

        # scale to -1, 1
        color_points = (color_points/255.0) - 0.5

        model = PCA()
        model.fit(color_points)

        # get the greenest eigen vector
        def green_cmp_fctor(a, b):
            if (a[1]-(a[0]+a[2])/2.0) < (b[1]-(b[0]+b[2])/2.0):
                return 1
            if (a[1]-(a[0]+a[2])/2.0) > (b[1]-(b[0]+b[2])/2.0):
                return -1
            return 0

        eigen = sorted(model.components_, key=cmp_to_key(green_cmp_fctor))
        color_points = (color_points @ eigen[0])[:, None]
        reconstructed = color_points @ eigen[0][None, :] + model.mean_
        color_points = ((color_points+0.5)*255).astype(np.uint8)
        reconstructed = ((reconstructed+0.5)*255).astype(np.uint8)
        return color_points, reconstructed  
        #np.reshape(color_points, (len(frames), self.height, self.width))

    def _get_threshold(self, data):
        return np.percentile(data, 98.0)

    def _open_writer(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        ts = datetime.now().strftime("%Y%m%d-%H%M")
        print(f"Writing to: out/{ts}.mp4")
        self.writer = cv2.VideoWriter(f"out/{ts}.mp4", fourcc, 
                    self.fps, (self.width, self.height))

    def _close_writer(self):
       self.writer.release()
       self.writer = None

    def _write_frame(self, frame):
        self.writer.write(frame)

    def _write_frame_as_image(self, frame, path, name):
        cv2.imwrite(path+name+".jpg", frame)

    def _draw_on_frame(self, frame, contours):
        for contour in contours:
            color = (0, 255, 0)
            
            M = cv2.moments(contour)
            if M["m00"]==0: 
                continue
            center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
            if center is not None:
                cv2.circle(frame,(center[0],center[1]),4,color,-1)
            if contour is not None:
                cv2.drawContours(frame,[contour],-1,color,2)

        return frame

    def _write_annotation_notes(self, frameid, contours):
        with open(f"dataset/labels/all/{frameid}.txt", "w") as fptr:
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)

                x_center = (x + w / 2) / self.width
                y_center = (y + h / 2) / self.height
                width = w / self.width
                height = h / self.height

                fptr.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def main():
    p_proc = PlantProcessor("videos/20260121_comp_2221_2114.mp4", batch_size=100)
    return p_proc.process()

if __name__ == "__main__":
    main()
