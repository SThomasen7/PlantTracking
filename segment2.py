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
    def __init__(self, contour):

        self.id = -1
        self.start_frame = -1
        self.end_frame = -1

        self.centers = list()
        self.shapes = list()
        self.areas = list()
        
        self.isValid = False

        self.addFrame(contour)

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
        live_count = 0
        for cnt in self.shapes:
            if cnt is not None:
                live_count += 1
        return live_count > 10

    def draw_on_frame(self, frame, frame_start, frame_id):
        idx = frame_start - self.start_frame + frame_id
        print("Frame start: ", self.start_frame, " end frame: ", self.end_frame)
        print("Frame offset: ", frame_start)
        print("current frame: ", frame_id)
        print(idx, "# frames ", len(self.centers))
        print("****")
        center = self.centers[idx]
        contour = self.shapes[idx]
        is_valid = self.is_valid()
        if is_valid:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        
        if center is not None:
            cv2.circle(frame,(center[0],center[1]),4,color,-1)
            cv2.putText(frame,f"{self.id}",(center[0],center[1]+5),
                        cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),2)
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
        self.objects = dict()
        self.live = list()
        self.dist_thresh = height*width*0.000005
        self.max_missing_frames = MISSING_FRAME_LIMIT
        print("Distance threshold: ", self.dist_thresh)
        self.new_frame()

    def add_bounding_shape(self, contour, frame=-1):
        self.frames[frame].append(contour)

    def new_frame(self):
        self.frames.append(list())

    def process_latest_frame(self):
        # get the live tracking objects
        live_objs = list()
        for live in self.live:
            live_objs.append(self.objects[live])

        sorted(live_objs, key=lambda o: o.id, reverse=True)

        contours = self.frames[-1]

        # if there are no live objects the contours are all added
        # as new live objects
        if len(live_objs) == 0:
            for cnt in contours:
                idx = len(self.objects)
                tobj = TrackingObject(cnt)
                tobj.id = idx
                tobj.start_frame = len(self.frames)
                self.objects[idx] = tobj
                self.live.append(idx)
            return

        # if there are live objects, match these contours to the nearest
        # live object
        dmat = np.ones((len(live_objs), len(contours))) * 999999

        status = dict()
        for i in range(len(live_objs)):
            status[i] = "pending"
            for j in range(len(contours)):
                dmat[i][j] = live_objs[i].dist(contours[j])


        # assign the contours to the closest live object
        taken = set()
        for i in range(len(live_objs)):
            dists = dmat[i]
            min_dist = dists.min()
            min_dist_idx = dists.argmin()
            
            if min_dist < self.dist_thresh:
                if min_dist_idx in taken:
                    live_objs[i].addFrame(None)
                    continue
                live_objs[i].addFrame(contours[min_dist_idx])
                taken.add(min_dist_idx)
            else:
                live_objs[i].addFrame(None)

        # create new live objects for the unassigned contours
        for j in range(len(contours)):
            if j in taken:
                continue
            idx = len(self.objects)
            tobj = TrackingObject(contours[j])
            tobj.id = idx
            tobj.start_frame = len(self.frames)
            self.objects[idx] = tobj
            self.live.append(idx)

        # kill the live objects that haven't had a match for 10 frames
        kill_set = set()
        for live in live_objs:
            if live.last_seen() >= self.max_missing_frames:
                kill_set.add(live.id)
                live.end_frame = len(self.frames) - live.last_seen()
                #print("!", live.start_frame, '->', live.end_frame, live.last_seen(), len(self.frames), len(live.shapes))
        print(self.live, kill_set)
        self.live = list(set(self.live) - kill_set)
        print(self.live)

    def draw_objects(self, frames, frame_start):
        for fidx in range(frame_start, frame_start + len(frames)):
            object_count = 0
            for obj in self.objects.values():
                for i in range(fidx):
                    if i >= obj.start_frame and obj.is_valid():
                        object_count += 1
                if fidx >= obj.start_frame and (fidx <= obj.end_frame or
                        obj.end_frame == -1):
                    print(len(frames), fidx, obj.start_frame)
                    obj.draw_on_frame(frames[fidx-frame_start], frame_start, fidx-frame_start)
            cv2.putText(frames[fidx-frame_start],f"Conteo: {object_count}",(30,50),
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
        self.fps = cap.get(cv2.CAP_PROP_FPS)
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
                if batch_id > 3:
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
    p_proc = PlantProcessor("videos/20260121_comp_2221_2114.mp4", batch_size=100)
    return p_proc.process()


if __name__ == "__main__":
    main()
