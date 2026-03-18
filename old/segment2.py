#!/bin/python3
import cv2
import numpy as np
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

class TrackingObject():
    def __init__(self, start_frame):

        self.id = -1
        self.start_frame = start_frame
        self.end_frame = -1

        self.centers = list()
        self.shapes = list()
        self.areas = list()
        
        self.count_id = -1
        #self.isValid = False
        #self.addFrame(contour)

    def addFrame(self, contour=None):
        if contour is None:
            self.centers.append(None)
            self.shapes.append(None)
            self.areas.append(None)
            return True
        elif type(contour) == type(list()):
            centers_x = list()
            centers_y = list()
            areas = list()
            for cnt in contour:
                M = cv2.moments(cnt)
                if M["m00"]==0: 
                    continue
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                centers_x.append(cx)
                centers_y.append(cy)
                areas.append(cv2.contourArea(contour))
            self.centers.append((
                np.array(centers_x).mean(),
                np.array(centers_y).mean()
            ))
            self.areas.append(np.array(areas).sum())
        else:
            self.shapes.append(contour)
            cx, cy = self._get_center(contour)
            if cx is None:
                self.centers.append(None)
                self.areas.append(None)
                return False

            self.centers.append((cx, cy))
            self.areas.append(cv2.contourArea(contour))
            return True

    def dist(self, cnt):
        center = self._get_center(cnt)
        if center[0] is None:
            return 999999
        missing_frames = 1
        for x in reversed(self.centers):
            if x is not None:
                dist = np.sqrt(
                        (x[0]-center[0])**2 +
                        (x[1]-center[1])**2
                )
                if missing_frames > MISSING_FRAME_LIMIT:
                    return 999999
                return dist / missing_frames
            else:
                missing_frames += 1
        return 999999

    def area(self):
        for x in reversed(self.areas):
            if x is not None:
                return x
        return -1

    def is_valid(self):
        return min([x[0] for x in self.centers]) < 1300

    def draw_on_frame(self, frames, frame_id):
        idx = frame_id - self.start_frame
        frame = frames[frame_id%len(frames)]

        center = self.centers[idx]
        contour = self.shapes[idx]
        is_valid = self.is_valid()

        if is_valid:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        
        if center is not None:
            cv2.circle(frame,(center[0],center[1]),4,color,-1)
            if is_valid:
                cv2.putText(frame,f"{self.count_id}",(center[0],center[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),2)
            cv2.putText(frame,f"{center[0]}, {center[1]}",(center[0]-10,center[1]+15),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
        if contour is not None:
            cv2.drawContours(frame,[contour],-1,color,2)


    def _get_center(self, cnt):
        M = cv2.moments(cnt)
        if M["m00"]==0: 
            self.centers.append(None)
            self.areas.append(None)
            return None, None
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])
        return cx, cy

    def last_seen(self):
        for i, contour in enumerate(reversed(self.shapes)):
            if contour is not None:
                return i+1
        return 999999

class Tracker():
    def __init__(self, height, width):
        self.frames = list()
        self.alive = list()
        self.objects = list()
        self.unique_count = list()
        self.valid = set()

        self.dist_thresh = height*width*0.00005
        self.max_missing_frames = MISSING_FRAME_LIMIT

        print("Distance threshold: ", self.dist_thresh)

        self.new_frame()

    def add_bounding_shape(self, contour, frame=-1):
        self.frames[frame].append(contour)

    def new_frame(self):
        self.frames.append(list())

    def process_latest_frame(self):
        # the first frame creates new objects
        if len(self.frames) == 1 or len(self.alive) == 0:
            for contour in self.frames[0]:
                obj = TrackingObject(len(self.frames)-1)
                obj.addFrame(contour)
                obj.id = len(self.objects)
                self.objects.append(obj)
                self.alive.append(obj)
                self.unique_count.append(len(self.alive))
            return

        if (len(self.frames[-1]) == 0):
            for obj in self.alive:
                obj.end_frame = len(self.frames)-1
            self.alive = list()
            return

        print("Frame: ", len(self.frames))
        # create the distance matrix for objects
        dist_mat = np.ones((len(self.alive), len(self.frames[-1])))*99999
        for i, obj in enumerate(self.alive):
            for j, contour in enumerate(self.frames[-1]):
                dist_mat[i][j] = obj.dist(contour)

        for obj in self.alive:
            print(obj.id, end=" ")
        print("")

        print(dist_mat)
        # until each object is assigned or the min distance is greater than
        # the threshold, assign objects

        min_dist = np.min(dist_mat)
        flat_index = np.argmin(dist_mat)
        row, col = np.unravel_index(flat_index, dist_mat.shape)

        assigned_objs = set()
        assigned_contours = set()
        while min_dist < self.dist_thresh and len(assigned_objs) < len(self.alive):

            # assign the object to that which is the closest
            self.alive[row].addFrame(self.frames[-1][col])
            assigned_objs.add(row)
            assigned_contours.add(col)
            dist_mat[row, :] = 99999

            min_dist = np.min(dist_mat)
            flat_index = np.argmin(dist_mat)
            row, col = np.unravel_index(flat_index, dist_mat.shape)

        print("Assigned objects: ", end="")
        for i in assigned_objs:
            print(f"({i}, {self.alive[i].id}) ", end="")
        print()

        print("Assigned contours: ", assigned_contours)
        # unassigned objects are no longer alive.
        # unassigned contours are no new objects
        for x in list(set(range(len(self.alive)))-assigned_objs):
            self.alive[x].end_frame = len(self.frames)-1

        new_alive = list()
        for x in list(assigned_objs):
            new_alive.append(self.alive[x])
        self.alive = new_alive

        new_object_c = 0
        for y in list(set(range(len(self.frames[-1])))-assigned_contours):
            obj = TrackingObject(len(self.frames)-1)
            obj.addFrame(self.frames[-1][y])
            obj.id = len(self.objects)

            self.objects.append(obj)
            self.alive.append(obj)
            new_object_c += 1

        self.unique_count.append(self.unique_count[-1]+new_object_c)

    def draw_objects(self, frames, frame_start):

        for fidx in range(frame_start, frame_start+len(frames)):
            for obj in self.objects:
                if fidx >= obj.start_frame and (fidx < obj.end_frame or obj.end_frame == -1):

                    if obj.is_valid() and obj.id not in self.valid:
                       self.valid.add(obj.id)
                       obj.count_id = len(self.valid)

                    obj.draw_on_frame(frames, fidx)

            cv2.putText(frames[fidx-frame_start],f"Conteo: {len(self.valid)}",(30,50),
                        cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),2)
                

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

        self.tracker = None

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
        self.tracker = Tracker(self.height, self.width)
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
                if batch_id > 6:
                    break
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
        MIN_AREA_RATIO = 0.0015
        min_area = MIN_AREA_RATIO*self.height*self.width

        for i in range(len(frames)):
            _, tmask = cv2.threshold(mask[i].astype(np.uint8), threshold, 255, cv2.THRESH_BINARY)
            contours,_ = cv2.findContours(tmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < min_area:
                    continue

                # if the neighborhood of the contour is not darker, assume a false positive
                #gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                #mask = np.zeros(gray.shape, dtype=np.uint8)
                #cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)

                #kernel = np.ones((7,7), np.uint8)
                #dilated = cv2.dilate(mask, kernel, iterations=1)
                #ring = cv2.subtract(dilated, mask)

                #inside_mean = cv2.mean(gray, mask=mask)[0]
                #ring_mean = cv2.mean(gray, mask=ring)[0]
                #difference = inside_mean - ring_mean
                # the immediate ring around the plant is lighter than the plant,
                # likely a false positive
                #if difference > 30:
                    #continue

                self.tracker.add_bounding_shape(cnt, -1)

                #centers.append((cx,cy))
                #cv2.circle(frames[i],(cx,cy),4,(0,0,255),-1)
                #cv2.drawContours(frames[i],[cnt],-1,(0,255,0),2)

                #self._write_frame(cv2.cvtColor(mask[i], cv2.COLOR_GRAY2BGR))

            self.tracker.process_latest_frame()
            self.tracker.new_frame()

        self.tracker.draw_objects(frames, start_idx)
        for i in range(len(frames)):
            self._write_frame(frames[i])

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
        # cluster into two groups get the lowest value of the higher cluster
        #model = KMeans(n_clusters=2)
        #labels = model.fit_predict(data)
        #means = [
            #data[labels == i].mean()
            #for i in range(2)
        #]
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


def main():
    p_proc = PlantProcessor("videos/linea_841.mp4", batch_size=100)
    return p_proc.process()


if __name__ == "__main__":
    main()
